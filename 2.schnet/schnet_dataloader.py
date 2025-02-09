from ovito_read_data_build_neigh_list import read_data_build_neigh_list, read_data_build_neigh_list_cutoff
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json

def list_label_files(directory, suffix):
    # List all files in the specified directory
    all_files = os.listdir(directory)
    # Filter files that end with ".label"
    label_files = [file for file in all_files if file.endswith(suffix)]
    return label_files

def calculate_h_diffusion_coeff(file_name):
	data = np.loadtxt(file_name, skiprows=2)
	x = data[:, 0]
	#y1 = data[:, 1]
	#y2 = data[:, 2]
	#y3 = data[:, 3]
	y4 = data[:, 4]
	y5 = y4[int(len(data)*0.1):]/x[int(len(data)*0.1):]/6*1e4
	return np.mean(y5)

def save_all_diffusion_coeffs(file_name, data_dict):
    with open(file_name, "w") as f:
        json.dump(data_dict, f)

def load_all_diffusion_coeffs(file_name):
    with open(file_name, "r") as f:
        data_dict = json.load(f)
    return data_dict

class CustomDataset(Dataset):
    def __init__(self, path, read3=False):
        """
        Args:
            path: path to folder containing lammps data file
        """
        self.all_file_names = os.listdir(path)
        self.file_names = list_label_files(path, ".equ.data")
        self.data_dicts = []
        self.labels = []
        self.has_all_diff_coeffs = False
        self.all_diffusion_coeffs = {}
        self.need_to_rebuild = False
        for name in self.all_file_names:
            if(name.endswith("all_diff_coeffs")):
                self.has_all_diff_coeffs = True
                self.all_diffusion_coeffs = load_all_diffusion_coeffs(path+name)
                break

        for idx, name in enumerate(self.file_names):
            #structure_types, r_ij, idx_i, idx_j = read_data_build_neigh_list(path+name)
            structure_types, r_ij, idx_i, idx_j, rmsd = read_data_build_neigh_list_cutoff(path+name)
            data_dict = {}
            data_dict["structure_types"] = torch.tensor(structure_types)
            data_dict["r_ij"] = torch.tensor(r_ij, dtype=torch.float32)
            data_dict["idx_i"] = torch.tensor(idx_i)
            data_dict["idx_j"] = torch.tensor(idx_j)
            data_dict["rmsd"] = torch.tensor(rmsd, dtype=torch.float32)
            data_dict["file_name"] = name
            self.data_dicts.append(data_dict)
            h_msd_file = name[:-9] + ".h_msd"
            if(self.has_all_diff_coeffs and name in self.all_diffusion_coeffs):
                diffusion_coeff = self.all_diffusion_coeffs[name]
            else:
                self.need_to_rebuild = True
                diffusion_coeff = calculate_h_diffusion_coeff(path+h_msd_file)
                if(read3):
                    h_msd_file = name[:-9] + ".h_msd2"
                    diffusion_coeff2 = calculate_h_diffusion_coeff(path+h_msd_file)
                    h_msd_file = name[:-9] + ".h_msd3"
                    diffusion_coeff3 = calculate_h_diffusion_coeff(path+h_msd_file)
                    diffusion_coeff += diffusion_coeff2
                    diffusion_coeff += diffusion_coeff3
                    diffusion_coeff /= 3.0
                self.all_diffusion_coeffs[name] = diffusion_coeff
            self.labels.append(diffusion_coeff)
            if((idx+1)%10==0 or (idx+1)==len(self.file_names)):
                print(f"loading {(idx+1)}/{len(self.file_names)}")
        if(not self.has_all_diff_coeffs or self.need_to_rebuild):
            print(f"building all_diff_coeffs dictionary")
            save_all_diffusion_coeffs(path+"all_diff_coeffs", self.all_diffusion_coeffs)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.file_names)

    def __getitem__(self, idx):
        """Return dict at idx"""
        return (self.data_dicts[idx], self.labels[idx])

