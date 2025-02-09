from schnet_model import AtomData2Scalar_mean
from schnet_dataloader import CustomDataset
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, random_split, ConcatDataset

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    excluded_keys = ["file_name"]
    for batch_data in train_loader:
        batch_targets_list = []
        batch_outputs_list = []
        for idx, datainfo in enumerate(batch_data):
            atom_info = datainfo[0]
            # Move the atom_info data to the GPU
            atom_info = {key: val.to(device) if key not in excluded_keys else torch.tensor(0).to(device) for key, val in atom_info.items()}
            
            # Append model outputs to the list (ensure it runs on GPU)
            batch_outputs_list.append(1.0 * model(atom_info))
            batch_targets_list.append(torch.log(torch.tensor(datainfo[1],dtype=torch.float32)))
        
        # Stack the list into a tensor (along the first dimension)
        batch_outputs = torch.stack(batch_outputs_list).to(device)
        batch_targets = torch.stack(batch_targets_list).to(device)
        # Ensure that batch_outputs requires gradients if needed
        batch_outputs.requires_grad_()

        # Compute loss
        loss = criterion(batch_outputs, batch_targets)
        running_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters
    return running_loss/len(train_loader)
        

# Function to validate the model
def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    excluded_keys = ["file_name"]
    with torch.no_grad():
        for batch_data in val_loader:
            batch_targets_list = []
            batch_outputs_list = []
            for idx, datainfo in enumerate(batch_data):
                atom_info = datainfo[0]
                # Move the atom_info data to the GPU
                atom_info = {key: val.to(device) if key not in excluded_keys else torch.tensor(0).to(device) for key, val in atom_info.items()}
                
                # Append model outputs to the list (ensure it runs on GPU)
                batch_outputs_list.append(1.0 * model(atom_info))
                batch_targets_list.append(torch.log(torch.tensor(datainfo[1],dtype=torch.float32)))
            
            # Stack the list into a tensor (along the first dimension)
            batch_outputs = torch.stack(batch_outputs_list).to(device)
            batch_targets = torch.stack(batch_targets_list).to(device)
            loss = criterion(batch_outputs, batch_targets)
            running_loss += loss.item()
    return running_loss / len(val_loader)

# Function to test the model
def test_model(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    all_outputs = []  # To store all model outputs
    all_targets = []  # To store all true target values
    running_loss = 0.0
    excluded_keys = ["file_name"]
    with torch.no_grad():  # Disable gradient calculation during testing
        for batch_data in test_loader:
            batch_targets_list = []  # List to collect targets for this batch
            batch_outputs_list = []  # List to collect model outputs for this batch
            
            for datainfo in batch_data:
                atom_info = datainfo[0]
                
                # Move atom_info and target data to GPU (or device)
                atom_info = {key: val.to(device) if key not in excluded_keys else torch.tensor(0).to(device) for key, val in atom_info.items()}
                target = torch.tensor(datainfo[1], dtype=torch.float32).to(device)
                
                # Get model output (forward pass) and append to list
                output = model(atom_info)
                batch_outputs_list.append(output)

                # Convert target values (log-transformation) and append
                batch_targets_list.append(torch.log(target))

            # Convert lists to tensors and accumulate for all batches
            all_outputs.append(torch.cat(batch_outputs_list))
            all_targets.append(torch.cat(batch_targets_list))
            
            # Calculate loss for this batch (optional, modify as needed)
            loss = criterion(torch.cat(batch_outputs_list), torch.cat(batch_targets_list))
            running_loss += loss.item()

    # Concatenate outputs and targets across all batches
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    
    return all_outputs, all_targets, running_loss / len(test_loader)

path = "../results/GB_hn10/"
#path = "./data4testrun/"
data_set = CustomDataset(path, read3=True)
path = "../results/bcc_T333_hn10"
bcc_data3 = CustomDataset(path, read3=False)

# Create DataLoader
#data_loader = DataLoader(data_set, batch_size=32, shuffle=True, collate_fn=lambda x: x)
train_size = 352
val_size = 128
test_size = len(data_set) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(data_set, [train_size, val_size, test_size])
train_dataset = ConcatDataset([train_dataset, bcc_data3])
# Create data loaders for train, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: x)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: x)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

for i in range(train_size):
    atom_dict, atom_label = train_dataset[i]
    file_name = atom_dict["file_name"]
    print(f"for training: {file_name}")

for i in range(val_size):
    atom_dict, atom_label = val_dataset[i]
    file_name = atom_dict["file_name"]
    print(f"for validation: {file_name}")

for i in range(test_size):
    atom_dict, atom_label = test_dataset[i]
    file_name = atom_dict["file_name"]
    print(f"for testing: {file_name}")

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create Model
model = AtomData2Scalar_mean().to(device)
# Define the loss function
criterion = nn.MSELoss()
# Define the Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Learning rate can be adjusted

# Training loop
num_epochs = 256
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    vali_loss = validate_model(model, val_loader, criterion, device)
    if((epoch+1)==128):
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
    if((epoch+1)%32==0):
        model_path = f'AtomData2Scalar_mean_hn10_train256_bcc_cutoff6_rmsd_ep{(epoch+1)}.pth'
        torch.save(model.state_dict(), model_path)
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Vali Loss: {vali_loss}')

# Save the model
model_path = f'AtomData2Scalar_mean_hn10_train256_bcc_cutoff6_rmsd_ep{(epoch+1)}.pth'
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')
