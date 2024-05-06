import os
import shutil
import random

# Define the paths
source_dir = r'C:\Users\naman\OneDrive\Desktop\fruit disease\pythonProject\3_pomegranate\pomegranate_disease_dataset'
train_dir = os.path.join(source_dir, 'pomegranate_disease_dataset/train')
test_dir = os.path.join(source_dir, 'pomegranate_disease_dataset/test')
valid_dir = os.path.join(source_dir, 'pomegranate_disease_dataset/valid')

# Create train, test, and valid directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# Define the subdirectories
subdirs = ['Alternaria', 'Anthracnose', 'Bacterial_Blight', 'Cercospora', 'Healthy']

# Define the split ratios
train_ratio = 0.7
test_ratio = 0.1
valid_ratio = 0.2


# Function to split data into train, test, and valid sets
def split_data(source, train, test, valid, split_ratios):
    for subdir in subdirs:
        subdir_path = os.path.join(source, subdir)
        files = os.listdir(subdir_path)
        random.shuffle(files)

        train_size = int(len(files) * split_ratios[0])
        test_size = int(len(files) * split_ratios[1])

        train_files = files[:train_size]
        test_files = files[train_size:train_size + test_size]
        valid_files = files[train_size + test_size:]

        # Copy files to train directory
        for file in train_files:
            src = os.path.join(subdir_path, file)
            dest = os.path.join(train, subdir)
            os.makedirs(dest, exist_ok=True)
            shutil.copy(src, dest)

        # Copy files to test directory
        for file in test_files:
            src = os.path.join(subdir_path, file)
            dest = os.path.join(test, subdir)
            os.makedirs(dest, exist_ok=True)
            shutil.copy(src, dest)

        # Copy files to valid directory
        for file in valid_files:
            src = os.path.join(subdir_path, file)
            dest = os.path.join(valid, subdir)
            os.makedirs(dest, exist_ok=True)
            shutil.copy(src, dest)


# Split data
split_data(source_dir, train_dir, test_dir, valid_dir, [train_ratio, test_ratio, valid_ratio])

print("Data splitting and directory structure creation complete.")
