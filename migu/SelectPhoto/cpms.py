from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

# Define transformations
transformations = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Specify your data directory
data_dir = "./beauty"

# Create your custom dataset instance
custom_dataset = CustomDataset(data_dir, transform=transformations)

# Create a data loader
batch_size = 10
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

print("The number of batches per epoch is: ", len(data_loader))



