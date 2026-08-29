import os
import random
import shutil

# Root directories
train_dir = "data/train"
val_dir = "data/val"

# Mapping: (train_subfolder, val_subfolder)
category_mapping = [
    ("FAKE", "ai"),
    ("REAL", "real")
]

split_ratio = 0.25  # 25% of train to validation

for train_sub, val_sub in category_mapping:
    source_folder = os.path.join(train_dir, train_sub)
    target_folder = os.path.join(val_dir, val_sub)
    
    # Ensure destination subfolder exists
    os.makedirs(target_folder, exist_ok=True)
    
    # Get all image filenames in train
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    # Calculate 25% (12,500 images)
    num_val = int(len(files) * split_ratio)
    
    # Randomly select files to move
    random.seed(42)  # Fixed seed for reproducibility
    val_files = random.sample(files, num_val)
    
    # Append files to existing val subfolders
    for file_name in val_files:
        src_path = os.path.join(source_folder, file_name)
        dst_path = os.path.join(target_folder, file_name)
        
        # Handle filename collisions if an image with the same name already exists in val
        if os.path.exists(dst_path):
            name, ext = os.path.splitext(file_name)
            dst_path = os.path.join(target_folder, f"{name}_train{ext}")
            
        shutil.move(src_path, dst_path)

    print(f"Appended {num_val} images from {source_folder} to {target_folder}")