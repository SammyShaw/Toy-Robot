# Toy Robot Image Preprocessing



pip install pillow-heif

from pillow_heif import register_heif_opener
from PIL import Image
import os 

# Working Directory
os.chdir("C:/Dat_Sci/Data Projects/Toy Image Data")

# Register HEIC support for Pillow
register_heif_opener()

 

#### CONVERT HEIC FILES TO JPGS ##########

dataset_path = "C:/Dat_Sci/Data Projects/Toy Image Data/"

for subdir, _, files in os.walk(dataset_path):
    for file in files:
        filepath = os.path.join(subdir, file)

        # Convert HEIC to JPG
        if file.lower().endswith(".heic"):
            img = Image.open(filepath)
            new_filepath = os.path.splitext(filepath)[0] + ".jpg"
            img.save(new_filepath, "JPEG")
            os.remove(filepath)  # Remove original HEIC file
            print(f"Converted {file} to JPG.")

print("Conversion complete!")


# Convert Data folders

dataset_path = "C:/Dat_Sci/Data Projects/Toy Robot/Data"

for subdir, _, files in os.walk(dataset_path):
    for file in files:
        filepath = os.path.join(subdir, file)

        # Convert HEIC to JPG
        if file.lower().endswith(".heic"):
            img = Image.open(filepath)
            new_filepath = os.path.splitext(filepath)[0] + ".jpg"
            img.save(new_filepath, "JPEG")
            os.remove(filepath)  # Remove original HEIC file
            print(f"Converted {file} to JPG.")

print("Conversion complete!")



#### RESIZE IMAGES #########################


# Path to dataset
dataset_path = "C:/Dat_Sci/Data Projects/Toy Robot/brios"
image_size = (128, 400)  # Target size for resizing

for split in ["training", "validation", "test"]:
    split_path = os.path.join(dataset_path, split)

    # ✅ Check if the split_path exists and is a directory
    if not os.path.isdir(split_path):
        print(f"Skipping {split_path} (not a directory)")
        continue

    for filename in os.listdir(split_path):
        img_path = os.path.join(split_path, filename)

        # ✅ Ensure it's a valid image file before processing
        if os.path.isfile(img_path):  
            try:
                print(f"Resizing: {img_path}")  # Debugging print

                with Image.open(img_path) as img:
                    img = img.resize(img_size)  # Resize image
                    img.save(img_path)  # Overwrite original file

            except Exception as e:
                print(f"Skipping {filename}: {e}")

print("Resizing complete!")



# For simple folder

for filename in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, filename)

        # ✅ Ensure it's a valid image file before processing
        if os.path.isfile(img_path):
            try:
                print(f"Resizing: {img_path}")  # Debugging print

                with Image.open(img_path) as img:
                    img = img.resize(image_size)  # Resize image
                    img.save(img_path)  # Overwrite original file

            except Exception as e:
                print(f"Skipping {filename}: {e}")

print("Resizing complete!")



#### Create Composite Image

pip install Pillow

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


os.chdir("C:/Dat_Sci/Data Projects/Toy Robot")


# Define paths
base_path = 'data/training'
categories = ['bananagrams', 'brios', 'cars', 'duplos', 'magnatiles']
images_per_category = 4  # Number of images per category

# Define image size and canvas size
img_size = (128, 128)
padding = 15
left_margin = padding
label_height = 50  # Space for labels at the top
canvas_width = (img_size[0] + padding) * len(categories) + left_margin
canvas_height = (img_size[1] + padding) * images_per_category + label_height

# Create a blank canvas
composite = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
draw = ImageDraw.Draw(composite)

# Define font for labels (using default if no TTF is specified)
try:
    font = ImageFont.truetype("arial.ttf", 20)  # You can change this to any .ttf font path
except:
    font = ImageFont.load_default()

# Add labels at the top
for col, category in enumerate(categories):
    x = col * (img_size[0] + padding) + img_size[0] // 2
    y = 10  # Position labels at the top
    text_width = draw.textbbox((0, 0), category, font=font)[2]  # Use textbbox to get text width
    draw.text((x - text_width // 2, y), category, font=font, fill=(0, 0, 0))


# Load and paste images
for col, category in enumerate(categories):
    folder = os.path.join(base_path, category)
    images = os.listdir(folder)[:images_per_category]
    
    for row, img_file in enumerate(images):
        img_path = os.path.join(folder, img_file)
        img = Image.open(img_path).resize(img_size)
        
        x = left_margin + col * (img_size[0] + padding) # Adjust for left margin
        y = row * (img_size[1] + padding) + label_height  # Adjust for label height
        composite.paste(img, (x, y))

# Save and display the composite image
composite.save('toy_collage.png')
plt.imshow(composite)
plt.axis('off')
plt.show()






from PIL import Image, ImageDraw, ImageFont
import os
import matplotlib.pyplot as plt

# Define paths and categories
base_path = 'data/training'
categories = ['bananagrams', 'brios', 'cars', 'duplos', 'magnatiles']
images_per_category = 4  # Number of images per category

# Define image size and canvas size (keep original settings)
img_size = (128, 128)
padding = 15
left_margin = padding  # Just add padding to the left margin
label_height = 50  # Space for labels at the top

# Adjust canvas size to add left margin only
canvas_width = (img_size[0] + padding) * len(categories) + left_margin
canvas_height = (img_size[1] + padding) * images_per_category + label_height

# Create a blank canvas
composite = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
draw = ImageDraw.Draw(composite)

# Define font for labels (using default if no TTF is specified)
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

# Add labels at the top with left margin padding
for col, category in enumerate(categories):
    x = left_margin + col * (img_size[0] + padding) + img_size[0] // 2
    y = 10  # Position labels at the top
    text_width = draw.textbbox((0, 0), category, font=font)[2]  # Use textbbox to get text width
    draw.text((x - text_width // 2, y), category, font=font, fill=(0, 0, 0))

# Load and paste images with left margin padding
for col, category in enumerate(categories):
    folder = os.path.join(base_path, category)
    images = os.listdir(folder)[:images_per_category]
    
    for row, img_file in enumerate(images):
        img_path = os.path.join(folder, img_file)
        img = Image.open(img_path).resize(img_size)
        
        x = left_margin + col * (img_size[0] + padding)  # Adjust for left margin only
        y = row * (img_size[1] + padding) + label_height  # Adjust for label height
        composite.paste(img, (x, y))

# Save and display the composite image
composite.save('toy_collage_with_padding.png', dpi=(300, 300))  # Save with high DPI
plt.figure(figsize=(12, 8), dpi=300)  # Ensure high-quality display
plt.imshow(composite)
plt.axis('off')
plt.show()





