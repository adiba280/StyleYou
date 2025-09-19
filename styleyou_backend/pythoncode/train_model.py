import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load the dataset CSV
csv_file = "C:/Users/muaaz/Downloads/styleyou/filtered_styles.csv"
df = pd.read_csv(csv_file)

print(f"Loaded dataset: {csv_file}, Total entries: {len(df)}")

# Extract necessary columns
df = df[['id', 'articleType']]  # Use id as filename, articleType as label
print("Columns selected: 'id' and 'articleType'")

# Convert labels to numeric values
label_map = {label: idx for idx, label in enumerate(df['articleType'].unique())}
df['label'] = df['articleType'].map(label_map)
print(f"Label mapping created: {label_map}")

# Split dataset into training and testing
train_data, test_data = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

# Define Dataset Class
class ClothingDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = str(self.dataframe.iloc[idx]['id']) + ".jpg"  # Assuming images are stored as 'id.jpg'
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Could not load image: {img_path}, Error: {e}")
            return None, None

        label = self.dataframe.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        print(f"[DEBUG] Loaded image: {img_path}, Label: {label}")
        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
image_folder = "C:/Users/muaaz/Downloads/archive/fashion-dataset/fashion-dataset/images2/"  # Change to your image folder path
train_dataset = ClothingDataset(train_data, image_folder, transform=transform)
test_dataset = ClothingDataset(test_data, image_folder, transform=transform)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("DataLoader initialized. Starting model setup...")

# Load Pretrained Model (ResNet18)
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(label_map))  # Adjust output layer to number of categories

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Model initialized on device: {device}")

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 5
print("Starting training...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    print(f"\nEpoch {epoch+1}/{num_epochs}:")

    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        if images is None or labels is None:
            print(f"[WARNING] Skipping batch {i} due to missing images.")
            continue

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 0:  # Print every 10 batches
            print(f"[DEBUG] Batch {i}: Loss = {loss.item():.4f}")

    print(f"Epoch {epoch+1} completed. Average Loss: {running_loss/len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "clothing_model.pth")
print("Model training complete and saved as clothing_model.pth!")

