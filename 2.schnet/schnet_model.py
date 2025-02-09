from typing import Callable, Dict, Union, Optional, List
import math
import torch
from torch import nn
from torch.nn import functional
import torch.nn.functional as F
from schnet_dense import Dense
#from schnet_dataloader import CustomDataset
from schnet_radial import GaussianRBF

def shifted_softplus(x: torch.Tensor):
    r"""Compute shifted soft-plus activation function.

    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (torch.Tensor): input tensor.

    Returns:
        torch.Tensor: shifted soft-plus of input.

    """
    return functional.softplus(x) - math.log(2.0)

class ShiftedSoftplus(torch.nn.Module):
    """
    Shifted softplus activation function with learnable feature-wise parameters:
    f(x) = alpha/beta * (softplus(beta*x) - log(2))
    softplus(x) = log(exp(x) + 1)
    For beta -> 0  : f(x) -> 0.5*alpha*x
    For beta -> inf: f(x) -> max(0, alpha*x)

    With learnable parameters alpha and beta, the shifted softplus function can
    become equivalent to ReLU (if alpha is equal 1 and beta approaches infinity) or to
    the identity function (if alpha is equal 2 and beta is equal 0).
    """

    def __init__(
        self,
        initial_alpha: float = 1.0,
        initial_beta: float = 1.0,
        trainable: bool = False,
    ) -> None:
        """
        Args:
            initial_alpha: Initial "scale" alpha of the softplus function.
            initial_beta: Initial "temperature" beta of the softplus function.
            trainable: If True, alpha and beta are trained during optimization.
        """
        super(ShiftedSoftplus, self).__init__()
        initial_alpha = torch.tensor(initial_alpha)
        initial_beta = torch.tensor(initial_beta)

        if trainable:
            self.alpha = torch.nn.Parameter(torch.FloatTensor([initial_alpha]))
            self.beta = torch.nn.Parameter(torch.FloatTensor([initial_beta]))
        else:
            self.register_buffer("alpha", initial_alpha)
            self.register_buffer("beta", initial_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate activation function given the input features x.
        num_features: Dimensions of feature space.

        Args:
            x (FloatTensor [:, num_features]): Input features.

        Returns:
            y (FloatTensor [:, num_features]): Activated features.
        """
        return self.alpha * torch.where(
            self.beta != 0,
            (torch.nn.functional.softplus(self.beta * x) - math.log(2)) / self.beta,
            0.5 * x,
        )

def scatter_add(
    x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    """
    Sum over values with the same indices.

    Args:
        x: input values
        idx_i: index of center atom i
        dim_size: size of the dimension after reduction
        dim: the dimension to reduce

    Returns:
        reduced input

    """
    return _scatter_add(x, idx_i, dim_size, dim)

@torch.jit.script
def _scatter_add(
    x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] = dim_size
    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    y = tmp.index_add(dim, idx_i, x)
    return y


def cosine_cutoff(input: torch.Tensor, cutoff: torch.Tensor):
    """ Behler-style cosine cutoff.

        .. math::
           f(r) = \begin{cases}
            0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
              & r < r_\text{cutoff} \\
            0 & r \geqslant r_\text{cutoff} \\
            \end{cases}

        Args:
            cutoff (float, optional): cutoff radius.

        """

    # Compute values of cutoff function
    input_cut = 0.5 * (torch.cos(input * math.pi / cutoff) + 1.0)
    # Remove contributions beyond the cutoff radius
    input_cut *= (input < cutoff).float()
    return input_cut

class CosineCutoff(nn.Module):
    r""" Behler-style cosine cutoff module.

    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    """

    def __init__(self, cutoff: float):
        """
        Args:
            cutoff (float, optional): cutoff radius.
        """
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, input: torch.Tensor):
        return cosine_cutoff(input, self.cutoff)

def replicate_module(
    module_factory: Callable[[], nn.Module], n: int, share_params: bool
):
    if share_params:
        module_list = nn.ModuleList([module_factory()] * n)
    else:
        module_list = nn.ModuleList([module_factory() for i in range(n)])
    return module_list


class SchNetInteraction(nn.Module):
    r"""SchNet interaction block for modeling interactions of atomistic systems."""

    def __init__(
        self,
        n_atom_basis: int,
        n_rbf: int,
        n_filters: int,
        activation: Callable = shifted_softplus,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            n_rbf (int): number of radial basis functions.
            n_filters: number of filters used in continuous-filter convolution.
            activation: if None, no activation function is used.
        """
        super(SchNetInteraction, self).__init__()
        self.in2f = Dense(n_atom_basis, n_filters, bias=False, activation=None)
        self.f2out = nn.Sequential(
            Dense(n_filters, n_atom_basis, activation=activation),
            Dense(n_atom_basis, n_atom_basis, activation=None),
        )
        self.filter_network = nn.Sequential(
            Dense(n_rbf, n_filters, activation=activation), Dense(n_filters, n_filters)
        )

    def forward(
        self,
        x: torch.Tensor,
        f_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        rcut_ij: torch.Tensor,
    ):
        """Compute interaction output.

        Args:
            x: input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j

        Returns:
            atom features after interaction
        """
        x = self.in2f(x)
        Wij = self.filter_network(f_ij)
        Wij = Wij * rcut_ij[:, None]

        # continuous-filter convolution
        x_j = x[idx_j]
        x_ij = x_j * Wij
        x = scatter_add(x_ij, idx_i, dim_size=x.shape[0])

        x = self.f2out(x)
        return x

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class GaussianFourierProjection_AtomDict(nn.Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(self, embed_dim, atom_key_name="charge", scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.atom_key_name = atom_key_name
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        if(self.atom_key_name in x.keys()):
            x = x[self.atom_key_name]
            x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
            x = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return x

class BinnedRMSDEmbedding_AtomDict(nn.Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(self, embed_dim, atom_key_name="rmsd", bin_size=100, bin_range=[0, 0.5]):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.atom_key_name = atom_key_name
        self.rmsd_bins = torch.linspace(bin_range[0], bin_range[1], bin_size-1)
        self.embedding = nn.Embedding(bin_size, embed_dim)
    def forward(self, x):
        if(self.atom_key_name in x.keys()):
            x = x[self.atom_key_name]
            device = x.device
            rmsd_bins = self.rmsd_bins.to(device)
            x = torch.bucketize(x, rmsd_bins, right=True)
            x = self.embedding(x)
        return x

class SchNet(nn.Module):
    """SchNet architecture for learning representations of atomistic systems

    References:

    .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
       Quantum-chemical insights from deep tensor neural networks.
       Nature Communications, 8, 13890. 2017.
    .. [#schnet_transfer] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
       SchNet: A continuous-filter convolutional neural network for modeling quantum
       interactions.
       In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
    .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.

    """

    def __init__(
        self,
        n_atom_basis: int = 64,
        n_interactions: int = 3,
        radial_basis: nn.Module = GaussianRBF(n_rbf=50, cutoff=5.0, trainable=False),
        cutoff_fn: nn.Module = CosineCutoff(cutoff=5.0),
        n_filters: int = None,
        shared_interactions: bool = False,
        activation: Union[Callable, nn.Module] = ShiftedSoftplus(),
        nuclear_embedding: Optional[nn.Module] = None,
        electronic_embeddings: Optional[List] = None,
        time_embedding: nn.Module = GaussianFourierProjection(embed_dim=64)
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            n_filters: number of filters used in continuous-filter convolution
            shared_interactions: if True, share the weights across
                interaction blocks and filter-generating networks.
            activation: activation function
            nuclear_embedding: custom nuclear embedding (e.g. spk.nn.embeddings.NuclearEmbedding)
            electronic_embeddings: list of electronic embeddings. E.g. for spin and
                charge (see spk.nn.embeddings.ElectronicEmbedding)
        """
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.n_filters = n_filters or self.n_atom_basis
        self.radial_basis = radial_basis
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff

        # initialize embeddings
        if nuclear_embedding is None:
            nuclear_embedding = nn.Embedding(100, n_atom_basis)
        self.embedding = nuclear_embedding
        if electronic_embeddings is None:
            electronic_embeddings = []
        electronic_embeddings = nn.ModuleList(electronic_embeddings)

        self.electronic_embeddings = electronic_embeddings

        # initialize interaction blocks
        self.interactions = replicate_module(
            lambda: SchNetInteraction(
                n_atom_basis=self.n_atom_basis,
                n_rbf=self.radial_basis.n_rbf,
                n_filters=self.n_filters,
                activation=activation,
            ),
            n_interactions,
            shared_interactions,
        )
        # initialize an embedding layer for timestep
        self.time_embedding = nn.Sequential(time_embedding, nn.Linear(n_atom_basis, n_atom_basis))

    def forward(self, inputs: Dict[str, torch.Tensor]):

        # get tensors from input dictionary
        atomic_numbers = inputs["structure_types"]
        r_ij = inputs["r_ij"]
        idx_i = inputs["idx_i"]
        idx_j = inputs["idx_j"]

        # compute pair features
        d_ij = torch.norm(r_ij, dim=1)
        f_ij = self.radial_basis(d_ij)
        rcut_ij = self.cutoff_fn(d_ij)

        # compute initial embeddings
        x = self.embedding(atomic_numbers)
        for embedding in self.electronic_embeddings:
            #x = x + embedding(x, inputs) 
            x = x + embedding(inputs)

        if("time_step" in inputs.keys()):
            time_step = inputs["time_step"]
            x = x + self.time_embedding(time_step)

        # compute interaction blocks and update atomic embeddings
        for interaction in self.interactions:
            v = interaction(x, f_ij, idx_i, idx_j, rcut_ij)
            x = x + v

        # collect results
        inputs["scalar_representation"] = x

        return inputs

class AtomWiseOutput(nn.Module):
    """
    MLPs acted on atom-wise representations
    """
    def __init__(
        self,
        n_atom_basis: int = 64,
        hidden_layers: List[int] = [32],
        n_out_basis: int = 16,
        activation: Callable = ShiftedSoftplus(),
    ):
        super().__init__()
        # Input layer
        self.input_layer = nn.Sequential(nn.Linear(n_atom_basis, hidden_layers[0]), activation)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for idx, layer_size in enumerate(hidden_layers[:-1]):
            next_layer_size = hidden_layers[idx + 1]
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(layer_size, next_layer_size),
                    activation
                )
            )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_layers[-1], n_out_basis)


    def forward(
        self,
        inputs: Dict[str, torch.Tensor]
    ):
        x = inputs["scalar_representation"]
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        inputs["scalar_representation"] = x
        return inputs

class AtomData2Scalar_mean(nn.Module):
    """
    Given preprocessed data of atoms
    return a scalar value
    """
    def __init__(
        self,
    ):
        super().__init__()
        # Input layer
        self.input_layer = SchNet(
            n_atom_basis = 64,
            n_interactions = 3,
            radial_basis = GaussianRBF(n_rbf=300, cutoff=6.0, trainable=False),
            cutoff_fn = CosineCutoff(cutoff=6.0),
            n_filters = None,
            shared_interactions = False,
            activation = ShiftedSoftplus(),
            nuclear_embedding = None,
            electronic_embeddings = [
                GaussianFourierProjection_AtomDict(embed_dim=64, atom_key_name="rmsd"),
                #BinnedRMSDEmbedding_AtomDict(embed_dim=64, atom_key_name="rmsd", bin_size=100, bin_range=[0, 0.49]) 
                ],
            time_embedding = GaussianFourierProjection(embed_dim=64)
            )
        
        # Atom wise layer for down sampling
        self.output_layer = AtomWiseOutput(
            n_atom_basis=64, 
            hidden_layers=[32], 
            n_out_basis=1 
            #activation=nn.Sigmoid()
            )
        
        # sum pooling will be appended atom-wise in forward()

    def forward(
        self,
        inputs: Dict[str, torch.Tensor]
    ):
        x = self.input_layer(inputs)
        x = self.output_layer(x)
        # average pooling
        x = torch.mean(x["scalar_representation"])
        return x
