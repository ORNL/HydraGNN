import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pickle
import h5py
import numpy as np
import time
start_time = time.time()
# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device available:", device)
class CustomImageDataset(Dataset):
    def __init__(self, images, features, labels):
        self.images = images
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        feature = torch.tensor(self.features[idx], dtype=torch.float32)  # Scalar feature
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, feature, label


class CNNEncoder(nn.Module):
    """
    A CNN model that takes an image and an extra scalar feature as input,
    then concatenates them for the regression task.
    """
    def __init__(self, image_size=(1, 256, 256)):
        super(CNNEncoder, self).__init__()
        self.image_size = image_size
        
        self.conv1 = nn.Conv2d(in_channels=self.image_size[0], out_channels=4, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.linear_line_size = int(16 * (image_size[1] // 4) * (image_size[2] // 4))
        
        self.fc1 = nn.Linear(in_features=self.linear_line_size + 1, out_features=128)  # +1 for the scalar feature
        self.fc2 = nn.Linear(in_features=128, out_features=4)

    def forward(self, x, scalar_feature):
        """
        Passes the image and scalar feature through the network.
        """
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        
        x = x.view(-1, self.linear_line_size)
        
        scalar_feature = scalar_feature.view(-1, 1) 
        x = torch.cat((x, scalar_feature), dim=1)
        
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        
        return x


image_shape = (1, 256, 256)
model = CNNEncoder(image_shape).to(device)
print(model)
h5_file = h5py.File('../pna_knn/MICRO2D_homogenized.h5', 'r')
VoidSmall = h5_file["VoidSmall"]

exclude_indices = [
    1434, 2229, 3013, 4921, 5502, 5670, 5956, 5989, 6250, 6270, 7425, 8099, 8294, 8361, 9358]

binary_images = []
print('Step prinitng')
for i in range(len(VoidSmall["VoidSmall"])):
    if i not in exclude_indices:
        # Get image and append to list
        binary_images.append(VoidSmall["VoidSmall"][i])
repeated_images = []
for img in binary_images:
    repeated_images.extend([img] * 6)
images=repeated_images
with open('VoidSmall_data_C_coefs_diff.pkl', 'rb') as f:
    data = pickle.load(f)
flattened_data = np.vstack([np.array(item['C_coefs_diff']) for item in data])
# flattened_datacrs = np.log(np.hstack([np.array(item['contrast_ratios']) for item in data]))

contrast_ratios =  (np.array([ -1., -2., -3.,  1.,  2.,  3.]) )  #np.array([ 1.,  2.,  3., -1., -2., -3.])
flattened_datacrs = (np.hstack([contrast_ratios for i in range(len(data))]))

labels = flattened_data  
feats=flattened_datacrs
train_images = np.array(images[:int(len(images) * 0.9)])
train_feats = np.array(feats[:int(len(images) * 0.9)])
train_labels = np.array(labels[:int(len(labels) * 0.9)])
val_images = np.array(images[int(len(images) * 0.9):])
val_feats = np.array(feats[:int(len(images) * 0.9)])
val_labels = np.array(labels[int(len(labels) * 0.9):])
train_images=train_images.reshape((-1,1,256,256))
val_images=val_images.reshape((-1,1,256,256))

print(np.shape(train_images))
print(np.shape(train_labels))

train_dataset = CustomImageDataset(train_images, train_feats, train_labels)
val_dataset = CustomImageDataset(val_images, val_feats, val_labels)
train_loader = DataLoader(train_dataset, batch_size=64)
val_loader = DataLoader(val_dataset, batch_size=64)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=200):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, features, labels in train_loader:

            images, features, labels = images.to(device), features.to(device), labels.to(device)
            
            outputs = model(images, features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Save the model every 250 epochs
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f"cnn_encoder_epochfeat_diff_{epoch + 1}.pt")
            print(f"Model saved at epoch {epoch + 1}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, features, labels in val_loader:

                images, features, labels = images.to(device), features.to(device), labels.to(device)
                outputs = model(images, features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)
# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=200)
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time:.4f} seconds")