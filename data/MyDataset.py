from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import pandas as pd


# Define custom dataset
class MyDataset(Dataset):
    def __init__(self, csv_file, img_dir, class_mapping, transform=None):
        """
        Initializes the dataset.
        
        :param csv_file: Path to the CSV file containing data.
        :param img_dir: Directory where images are stored.
        :param class_mapping: Dictionary mapping class names to numeric values.
        :param transform: Optional transform to be applied on a sample.
        """
        data_frame = pd.read_csv(csv_file)
        self.image_names = data_frame.iloc[:, 0]  # Assuming image name is in the first column
        self.csv_data = data_frame.iloc[:, 2:]  # Assuming other data start from the 3rd column
        self.targets = data_frame.iloc[:, 1]  # Assuming the 2nd column is the target
        self.img_dir = img_dir
        self.transform = transform
        self.class_mapping = class_mapping

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        csv_data_row = self.csv_data.iloc[idx]
        csv = torch.tensor(csv_data_row.values, dtype=torch.float)
        
        # Map the target using the provided class_mapping dictionary
        target = torch.tensor(self.class_mapping.get(self.targets[idx], -1), dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return (image, csv), target
