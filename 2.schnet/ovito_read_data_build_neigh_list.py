from ovito.io import import_file, export_file
from ovito.data import *
from ovito.modifiers import *
import numpy as np

def read_data_build_neigh_list(fileName, neigh_num=64):
    pipeline = import_file(fileName)
    pipeline.modifiers.append(PolyhedralTemplateMatchingModifier(output_rmsd=True))
    data = pipeline.compute()
    finder = NearestNeighborFinder(neigh_num, data)

    # neigh_idx: (M, N) indices of neighbor particles, 
    # M equal to the total number of particles 
    # N refers to the number of nearest neighbors 
    # neigh_vec: (M, N, 3) for x-, y- and z- components of delta
    neigh_idx, neigh_vec = finder.find_all()

    # M is the total number of particles, N is the number of nearest neighbors
    M, N = neigh_idx.shape   
    # idx_i: (M*N) array of all i- (centered) atoms
    # Repeat each particle index N times (since each particle has N neighbors)
    idx_i = np.repeat(np.arange(M), N)   
    # idx_j: (M*N) array of all j- (neighbor) atoms
    # Flatten the neighbor indices
    idx_j = neigh_idx.flatten()
    # r_ij: (M*N, 3) array of x-, y- and z- component of delta of each pair
    # Reshape the neighbor vectors into a (M*N, 3) array
    r_ij = neigh_vec.reshape(-1, 3)
    # Now, idx_i contains the indices of the center particles,
    # idx_j contains the indices of the neighbors,
    # r_ij contains the delta vectors for each particle-neighbor pair.
    rmsd = np.array(data.particles['RMSD'])
    #rmsd_bins = np.linspace(0,0.495,99)
    rmsd_bins = np.linspace(0,0.196,99)
    structure_types = np.digitize(rmsd, rmsd_bins, right=True)
    return structure_types, r_ij, idx_i, idx_j


def read_data_build_neigh_list_cutoff(fileName, cutoff=6.0):
    pipeline = import_file(fileName)
    pipeline.modifiers.append(PolyhedralTemplateMatchingModifier(output_rmsd=True))
    data = pipeline.compute()
    finder = CutoffNeighborFinder(cutoff, data)

    # neigh_idx: (M*N, 2) indices of neighbor particles, 
    # M equal to the total number of particles 
    # N refers to the number of nearest neighbors 
    # neigh_vec: (M*N, 3) for x-, y- and z- components of delta
    neigh_idx, neigh_vec = finder.find_all()
    # idx_i: (M*N) array of all i- (centered) atoms
    idx_i = neigh_idx[:, 0]   
    # idx_j: (M*N) array of all j- (neighbor) atoms
    idx_j = neigh_idx[:, 1]
    # r_ij: (M*N, 3) array of x-, y- and z- component of delta of each pair
    r_ij = neigh_vec
    # Now, idx_i contains the indices of the center particles,
    # idx_j contains the indices of the neighbors,
    # r_ij contains the delta vectors for each particle-neighbor pair.
    rmsd = np.array(data.particles['RMSD'])
    #rmsd_bins = np.linspace(0,0.49,99)
    #structure_types = np.digitize(rmsd, rmsd_bins, right=True)
    structure_types = data.particles['Structure Type']
    return structure_types, r_ij, idx_i, idx_j, rmsd
#path = "../results/GB_hn1/"
#name = "box28gb1hn1r1x-0.014r1y0.690r1z-0.723r1d185r2x-0.675r2y0.566r2z0.472r2d54.equ.data"
#
#atom_numbers, r_ij, idx_i, idx_j = read_data_build_neigh_list(path+name)
#
#print(np.sum(atom_numbers))
