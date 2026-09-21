import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.onnx
from generate_data import generate_dataset

# 1. Define the Generic CNN Architecture (Used for both Foot and Waist)
class CricketClassifier(nn.Module):
    def __init__(self):
        super(CricketClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return x

def get_dataset_path(project_backend, raw_folder_name, mock_folder_name):
    raw_dir = os.path.join(project_backend, "data", raw_folder_name)
    mock_dir = os.path.join(project_backend, "data", mock_folder_name)
    
    if os.path.exists(raw_dir) and len(os.listdir(raw_dir)) >= 2:
        # Check subfolders
        subfolders = os.listdir(raw_dir)
        total_files = sum(len(os.listdir(os.path.join(raw_dir, f))) for f in subfolders if os.path.isdir(os.path.join(raw_dir, f)))
        if total_files > 20:
            return raw_dir
            
    # Fallback to mock
    if not os.path.exists(mock_dir) or len(os.listdir(mock_dir)) < 2:
        print(f"Data missing for {mock_folder_name}. Generating synthetic datasets...")
        generate_dataset(num_samples_per_class=500, 
                         base_dir=os.path.join(project_backend, "data", "mock"),
                         waist_dir=os.path.join(project_backend, "data", "waist_mock"))
    return mock_dir

def train_generic_model(data_dir, model_filename, epochs=10, batch_size=32, lr=0.001):
    project_backend = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"\n--- Training Model: {model_filename} ---")
    print(f"Loading data from: {data_dir}")
    
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    print(f"Classes found: {full_dataset.classes}")
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CricketClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float("inf")
    models_dir = os.path.join(project_backend, "models")
    os.makedirs(models_dir, exist_ok=True)
    pth_path = os.path.join(models_dir, f"{model_filename}.pth")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = correct_train / total_train
        
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = correct_val / total_val
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.2f}%")
              
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), pth_path)
            
    print(f"Best weights saved to: {pth_path}")
    export_to_onnx(pth_path, project_backend, model_filename)

def export_to_onnx(pth_path, project_backend, model_filename):
    onnx_path = os.path.join(project_backend, "models", f"{model_filename}.onnx")
    model = CricketClassifier()
    model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
    model.eval()
    dummy_input = torch.randn(1, 3, 128, 128)
    
    try:
        torch.onnx.export(
            model, dummy_input, onnx_path, export_params=True,
            opset_version=18, do_constant_folding=True,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"ONNX model saved to: {onnx_path}")
    except Exception as e:
        print(f"ONNX export skipped/failed: {e}")

def train_all_models():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_backend = os.path.dirname(script_dir)
    
    # 1. Train Crease Model (Foot overstepping)
    crease_data_dir = get_dataset_path(project_backend, "raw_images", "mock")
    train_generic_model(crease_data_dir, "crease_classifier", epochs=10)
    
    # 2. Train Waist Model (Beamers)
    waist_data_dir = get_dataset_path(project_backend, "raw_waist_images", "waist_mock")
    train_generic_model(waist_data_dir, "waist_classifier", epochs=10)

if __name__ == "__main__":
    train_all_models()
