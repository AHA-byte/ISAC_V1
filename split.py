import os
import random
import shutil

def split_dataset(dataset_path, split_ratio=0.2):
    # Setup the paths based on CVAT's export structure
    train_img_dir = os.path.join(dataset_path, 'train', 'images')
    train_lbl_dir = os.path.join(dataset_path, 'train', 'labels')
    
    val_img_dir = os.path.join(dataset_path, 'valid', 'images')
    val_lbl_dir = os.path.join(dataset_path, 'valid', 'labels')

    # Create the validation directories if they don't exist
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    # Grab all the images from the train folder
    images = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.png'))]

    # Calculate exactly how many images make up 20%
    num_val = int(len(images) * split_ratio)

    # Randomly select the images to move
    val_images = random.sample(images, num_val)

    print(f"Randomly moving {num_val} images to the validation set...")

    for img in val_images:
        # Define the matching text file name
        lbl = img.rsplit('.', 1)[0] + '.txt'

        # Move the image
        shutil.move(os.path.join(train_img_dir, img), os.path.join(val_img_dir, img))

        # Move the label (includes a safety check for empty background images)
        lbl_path = os.path.join(train_lbl_dir, lbl)
        if os.path.exists(lbl_path):
            shutil.move(lbl_path, os.path.join(val_lbl_dir, lbl))

    print("Split complete! Your dataset is ready for training.")

if __name__ == '__main__':
    # REPLACE THIS with the actual name of your unzipped CVAT folder
    split_dataset('ISAC_V1_dataset_folder_name')