import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import direction_utils as utils

learning_rate=1e-4
num_epochs = 1000
patience = 30  # Number of epochs to wait before stopping if no improvement
min_delta = 1e-3 # Minimum change to qualify as improvement
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
torch.cuda.manual_seed(0) 
np.random.seed(0)


def model_evaluation(model, val_loader):
    model.eval()
    # Loss and Accuracy
    val_loss = 0.0
    val_acc = 0.0

    criterion = nn.CrossEntropyLoss().to(device)  # Move loss function to GPU if needed
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            se_electrode1 = model.electrode_weights_layer1[-1].tolist()
            se_electrode2 = model.electrode_weights_layer2.cpu().tolist()

            se1_scales = model.se1_scales.cpu().tolist()
            se2_scales = model.se2_scales.cpu().tolist()
            se3_scales = model.se3_scales.cpu().tolist()
    
            loss = criterion(outputs, targets.squeeze())
            val_loss += loss.item()
            val_acc += calculate_accuracy(outputs, targets.squeeze())

    # Calculate average validation loss for this epoch
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    assigned_ranks = {'electrodes_se1':se_electrode1, 
                      'electrodes_se2':se_electrode2,
                      'filter_se1': se1_scales, 
                      'filter_se2': se2_scales, 
                      'filter_se3': se3_scales 
                      }
    

    return val_loss, val_acc, assigned_ranks

def model_training(model, train_loader, val_loader, Tuning=False, verbose=False):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)  # Move loss function to GPU if needed

    # Training with Early Stopping
    best_val_acc = 0
    early_stop_counter = 0

    # Placeholder for training and validation loss history
    train_losses = []
    val_losses = []
    val_accuracy = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.squeeze())

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # val_loss, val_acc, el_rank1, el_rank2 = model_evaluation(model, val_loader)
        val_loss, val_acc, _ = model_evaluation(model, val_loader)

        val_losses.append(val_loss)
        val_accuracy.append(val_acc)
        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}%')
        
        if Tuning:
            # Early Stopping
            # Early stopping check
            if val_acc > best_val_acc + min_delta:  # Check if validation Acc improved
                best_val_acc = val_acc
                early_stop_counter = 0  # Reset early stop counter
                torch.save(model.state_dict(), 'trained_model_checkpoint.pth')  # Save best model
                if verbose:
                    print(f'New best validation accuracy: {best_val_acc:.4f} at epoch {epoch + 1}')

            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch + 1} with best validation accuracy: {best_val_acc:.4f}')
                break

        else:
            if epoch > 100:
                # Early stopping check
                if val_acc > best_val_acc + min_delta:  # Check if validation Acc improved
                    best_val_acc = val_acc
                    early_stop_counter = 0  # Reset early stop counter
                    torch.save(model.state_dict(), 'trained_model_checkpoint.pth')  # Save best model
                    if verbose:
                        print(f'New best validation accuracy: {best_val_acc:.4f} at epoch {epoch + 1}')

                else:
                    early_stop_counter += 1

                if early_stop_counter >= patience and best_val_acc > 70.00:
                    if verbose:
                        print(f'Early stopping at epoch {epoch + 1} with best validation accuracy: {best_val_acc:.4f}')
                    break
    
    return model
 
# Function to calculate accuracy
def calculate_accuracy(preds, labels):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy*100

def convert_to_tensor(X, Y, batch_size=32, shuffle=False):
    
    Xbase = utils.baseline_correction(X)
    Xfilt = utils.bandpass_filtering(Xbase)

    X_tensor = torch.tensor(Xfilt, dtype=torch.float32).unsqueeze(1).permute(0, 1, 3, 2).to(device)  # Shape: (batch_size, 1, 27, 2500)
    Y_tensor = torch.tensor(Y, dtype=torch.long).to(device)  # Use long for classification

    torch_data = TensorDataset(X_tensor, Y_tensor)
    data_loader = DataLoader(dataset=torch_data, batch_size=batch_size, shuffle=shuffle)


    return data_loader

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Output size: (batch_size, in_channels, 1, 1)
        
        # self.fc1 = nn.Linear(in_channels, max(1, in_channels // reduction), bias=False)  # Squeeze
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)  # Squeeze

        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)  # Excitation
        self.sigmoid = nn.Sigmoid()

        self.filter_scale = None

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Squeeze: Global Average Pooling
        out = self.global_avg_pool(x).view(batch_size, channels)  # Shape: (batch_size, in_channels)
        
        # Excitation: Fully connected layers
        out = self.fc1(out)  # Shape: (batch_size, in_channels // reduction)
        out = self.relu(out)
        out = self.fc2(out)  # Shape: (batch_size, in_channels)
        out = self.sigmoid(out)
        self.filter_scale = out.mean(dim=0, keepdim=True).detach().cpu()

        out = out.view(batch_size, channels, 1, 1)
        
        # out = self.sigmoid(out).view(batch_size, channels, 1, 1)  # Reshape to (batch_size, in_channels, 1, 1)
        
        # Scale the input by the SE weights
        return x * out.expand_as(x)
    
class SEBlockPerChannel(nn.Module):
    def __init__(self, height, width, reduction=16):
        super(SEBlockPerChannel, self).__init__()
        self.height = height
        self.width = width
        
        # Fully connected layers to learn the importance of each height (electrode) for each channel
        self.fc1 = nn.Linear(height, height // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(height // reduction, height, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.channel_weights = None

    def forward(self, x):
        # Input x shape: (batch_size, channels, height, width)
        b, c, h, w = x.size()
        
        # Initialize a list to store the reweighted channels
        output = []
        # print(f'Shape of X Before SE: {x.size()}')

        # Loop over each channel (for each filter)
        electrode_scale = []
        for i in range(c):
            # Extract the i-th channel (filter): shape (batch_size, height, width)
            y = x[:, i, :, :]
            # print(f'Shape of X for filter {i} SE: {y.size()}')

            # Squeeze: Average pooling along the width (temporal dimension)
            y_squeezed = y.mean(dim=-1)  # shape: (batch_size, height)
            # print(f"Shape before FC1 (y_squeezed): {y_squeezed.shape}")
            # print(f'Shape of X for filter {i} Pooling Layer SE: {y_squeezed.size()}')
            
            # Excitation: Fully connected layers to assign weights to the height dimension
            y_fc1 = self.fc1(y_squeezed)
            # print(f'Shape of X for filter {i} FC Layer1 SE: {y_fc1.size()}')
            y_relu = self.relu(y_fc1)

            y_fc2 = self.fc2(y_relu)
            # print(f'Shape of X for filter {i} FC Layer2 SE: {y_fc2.size()}')

            y_sigmoid = self.sigmoid(y_fc2)  # shape: (batch_size, height)

            electrode_scale.append(y_sigmoid.mean(dim=0, keepdim=True).detach().cpu())
            
            # Reshape to (batch_size, 1, height, 1) to broadcast
            y_sigmoid = y_sigmoid.view(b, 1, h, 1)
            # print(f'Shape of X for filter {i} Sigmoid SE: {y_sigmoid.size()}')
            
            # Scale the original channel data
            y_scaled = y.unsqueeze(1) * y_sigmoid.expand_as(y.unsqueeze(1))  # Reshape y to (batch_size, 1, height, width)
            # print(f'Shape of X for filter {i} After SE: {y_scaled.size()}')
            
            # Append to output list
            output.append(y_scaled)
        
        # Concatenate along the channel dimension to restore the original shape
        output = torch.cat(output, dim=1)  # shape: (batch_size, channels, height, width)
        # print(f'Shape of X for After SE: {output.size()}')
        self.channel_weights = torch.cat(electrode_scale, dim=0)
        
        return output

class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=27, Samples=2500, dropoutRate=0.5, 
                 kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet, self).__init__()
        
        # Handle dropout type
        if dropoutType == 'SpatialDropout2D':
            self.dropout = nn.Dropout2d(dropoutRate)
        elif dropoutType == 'Dropout':
            self.dropout = nn.Dropout(dropoutRate)
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout.')

        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        # Squeeze-and-Excitation Block
        self.se_electrode1 = SEBlockPerChannel(Chans, Samples, reduction=3)
        self.se_electrode2 = SEBlockPerChannel(Chans, Samples, reduction=3)
        self.se1 = SEBlock(F1, 3)


        self.depthwiseConv = nn.Conv2d(F1, F1*D, (Chans, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1*D)
        self.se2 = SEBlock(F1*D, 3)  # Squeeze-and-Excitation Block
        self.pool1 = nn.AvgPool2d((1, 4))

        # Block 2
        self.separableConv = nn.Conv2d(F1*D, F2, (1, 16), padding='same', bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.se3 = SEBlock(F2, 3)  # Squeeze-and-Excitation Block
        self.pool2 = nn.AvgPool2d((1, 8))

        # Flatten and Dense
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(F2 * (Samples // (4 * 8)), nb_classes)
        self.norm_constraint = nn.utils.weight_norm(self.dense)
        
        #For Debuggin and Analysis
        self.electrode_weights_layer1 = None
        self.electrode_weights_layer2 = None
        self.se1_scales = None
        self.se2_scales = None
        self.se3_scales = None
        
    def forward(self, x):
        # Block 1
        x = self.se_electrode1(x)
        self.electrode_weights_layer1 = self.se_electrode1.channel_weights
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.se_electrode2(x)
        self.electrode_weights_layer2 = self.se_electrode2.channel_weights
        x = self.se1(x)
        self.se1_scales = self.se1.filter_scale
        x = self.depthwiseConv(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.se2(x)
        self.se2_scales = self.se2.filter_scale
        x = self.pool1(x)
        x = self.dropout(x)

        # Block 2
        x = self.separableConv(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.se3(x)
        self.se3_scales = self.se3.filter_scale
        x = self.pool2(x)
        x = self.dropout(x)

        # Flatten and Dense
        x = self.flatten(x)
        x = self.dense(x)
        return F.softmax(x, dim=1)