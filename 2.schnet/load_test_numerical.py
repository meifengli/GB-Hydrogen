from schnet_model import AtomData2Scalar_mean
from schnet_dataloader import CustomDataset
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import re
import numpy as np

def process_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = {
        'training': [],
        'validation': [],
        'testing': [],
        # Add more categories if needed
    }

    for line in lines:
        # Match the pattern and extract type and file name
        match = re.match(r'for (\w+): (.+)', line)
        if match:
            data_type = match.group(1)
            file_name = match.group(2)
            if data_type in data:
                data[data_type].append(file_name)
            else:
                data[data_type] = [file_name]

    return data

train_output = "train_cutoff5_rmsd.output"
file_names_data = process_lines(train_output)

path = "../../results/GB_hn10/"
data_set = CustomDataset(path)
#path = "../../results/GB_hn10/"
#data_set = CustomDataset(path, read3=True)
train_set = [data_set[i] for i in range(len(data_set)) if data_set[i][0]["file_name"] in file_names_data["training"] ]
vali_set = [data_set[i] for i in range(len(data_set)) if data_set[i][0]["file_name"] in file_names_data["validation"] ]
test_set = [data_set[i] for i in range(len(data_set)) if data_set[i][0]["file_name"] in file_names_data["testing"] ]

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create Model
model = AtomData2Scalar_mean().to(device)
model_path = 'AtomData2Scalar_mean_hn10_train256_bcc_cutoff6_rmsd_ep256.pth'
# Load the saved state dictionary
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode
excluded_keys = ["file_name"]
y_gt = []
y_pred = []
for datainfo in train_set:
	atom_info = datainfo[0]
	atom_info_to = {key: val.to(device) if key not in excluded_keys else torch.tensor(0).to(device) for key, val in atom_info.items()}
	y_gt.append(np.log(datainfo[1]))
	y_pred.append(model(atom_info_to).detach().cpu())
	print(atom_info["file_name"], y_gt[-1], y_pred[-1].numpy())

for datainfo in vali_set:
	atom_info = datainfo[0]
	atom_info_to = {key: val.to(device) if key not in excluded_keys else torch.tensor(0).to(device) for key, val in atom_info.items()}
	y_gt.append(np.log(datainfo[1]))
	y_pred.append(model(atom_info_to).detach().cpu())
	print(atom_info["file_name"], y_gt[-1], y_pred[-1].numpy())

for datainfo in test_set:
	atom_info = datainfo[0]
	atom_info_to = {key: val.to(device) if key not in excluded_keys else torch.tensor(0).to(device) for key, val in atom_info.items()}
	y_gt.append(np.log(datainfo[1]))
	y_pred.append(model(atom_info_to).detach().cpu())
	print(atom_info["file_name"], y_gt[-1], y_pred[-1].numpy())


