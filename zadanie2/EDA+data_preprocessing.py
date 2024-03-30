import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

pic = "consep_1" # input("Enter the picture name:")

label = sio.loadmat(f'lizard_labels/Lizard_Labels/Labels/{pic}.mat')

# Load the instance segmentation map.
# This map is of type int32 and contains values from 0 to N, where 0 is background
# and N is the number of nuclei.
# Shape: (H, W) where H and W is the height and width of the image.
inst_map = label['inst_map']
print(inst_map.shape)


# Load the index array. This determines the mapping between the nuclei in the instance map and the
# corresponing provided categories, bounding boxes and centroids.
nuclei_id = label['id'] # shape (N, 1), where N is the number of nuclei.

# Load the nuclear categories / classes.
# Shape: (N, 1), where N is the number of nuclei.
classes = label['class']

# Load the bounding boxes.
# Shape: (N, 4), where N is the number of nuclei.
# For each row in the array, the ordering of coordinates is (y1, y2, x1, x2).
bboxs = label['bbox']

# Load the centroids.
# Shape: (N, 2), where N is the number of nuclei.
# For each row in the array, the ordering of coordinates is (x, y).
centroids = label['centroid']

# Matching each nucleus with its corresponding class, bbox and centroid:

# Get the unique values in the instance map - each value corresponds to a single nucleus.
unique_values = np.unique(inst_map).tolist()[1:] # remove 0

# print(inst_map, nuclei_id, classes, bboxs, centroids, unique_values)

# Convert nuclei_id to list.
nuclei_id = np.squeeze(nuclei_id).tolist()

# Initialize lists to store data
indices = []
class_list = []
bbox_list = []
centroid_list = []

for value in unique_values:
    idx = nuclei_id.index(value)  # Get the index
    indices.append(idx)
    class_list.append(classes[idx].item())  # Convert to Python int
    bbox_list.append(bboxs[idx].tolist())
    centroid_list.append(centroids[idx].tolist())

# Create a DataFrame
data = {'Index': indices, 'Class': class_list, 'Bounding Box': bbox_list, 'Centroid': centroid_list}
df = pd.DataFrame(data)

# Display the DataFrame
print(print(df.to_string(index=False)))

# Load the image
image_path = f'lizard_images1/Lizard_Images1/{pic}.png'
image = Image.open(image_path)

# Plot the image
plt.figure(figsize=(7, 8))
plt.imshow(image)
plt.title(f'{pic}.png')
plt.axis('on')
plt.show()

# Plot the heatmap
plt.figure(figsize=(8, 8))
plt.imshow(inst_map, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Nucleus ID')
plt.title('Instance Segmentation Heatmap')
plt.xlabel('Width')
plt.ylabel('Height')
plt.show()

# Define the number of unique classes
num_classes = len(df['Class'].unique())

# Create a colormap with a different color for each class
cmap = plt.cm.get_cmap('tab10', num_classes)

# Plot the heatmap
plt.figure(figsize=(8, 8))
plt.imshow(inst_map, cmap=cmap, interpolation='nearest')
plt.colorbar(label='Nucleus ID')
plt.title('Semantic Segmentation Heatmap')
plt.xlabel('Width')
plt.ylabel('Height')
plt.show()

# # Initialize a set to keep track of printed classes
# printed_classes = set()

# # Create a figure and axis
# fig, ax = plt.subplots(figsize=(8, 4))

# # Add class numbers and names to the plot
# for class_id, color in zip(range(1, num_classes + 1), cmap.colors):
#     # Check if the class has already been printed
#     if class_id not in printed_classes:
#         # Plot a bar with the class number
#         ax.bar(class_id, 1, color=color, label=f'Class {class_id}')
#         # Add the class to the set of printed classes
#         printed_classes.add(class_id)

# # Set labels and legend
# ax.set_xlabel('Class')
# ax.set_ylabel('Color')
# ax.set_title('Unique Classes and Colors')
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# # Remove x-axis ticks
# ax.set_xticks([])

plt.show()

import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            try:
                img = Image.open(img_path)
                images.append(img)
            except Exception as e:
                print(f"Error loading image '{img_path}': {e}")
    return images

def images_to_array(images, target_size=(500, 500)):
    num_images = len(images)
    if num_images == 0:
        return None

    array_shape = (num_images, *target_size, 3)  # Assuming RGB images
    image_array = np.zeros(array_shape, dtype=np.uint8)

    for i, img in enumerate(images):
        img = img.resize(target_size)  # Resize image
        image_array[i] = np.array(img)

    return image_array


folder_path = 'images_all'
save_file = 'images_array.npy'

# Check if the file exists
if os.path.exists(save_file):
    # If the file exists, load the array from the file
    X = np.load(save_file)
    print("Images array loaded from file.")
else:
    # If the file doesn't exist, load images from folder and convert to array
    images = load_images_from_folder(folder_path)
    X = images_to_array(images)

    # Save X to a file
    np.save(save_file, X)
    print("Images array saved to file.")

print("X shape:", X.shape)
del X


def load_index_class_pairs(pic):
    label = sio.loadmat(f'lizard_labels/Lizard_Labels/Labels/{pic}.mat')

    # Load the index array.
    nuclei_id = label['id'].squeeze().tolist()  # Convert to list

    # Load the nuclear categories / classes.
    classes = label['class'].squeeze().tolist()  # Convert to list

    # Combine index and class into a dictionary
    index_class_dict = {index: class_ for index, class_ in zip(nuclei_id, classes)}

    return index_class_dict

def replace_indices_with_classes(inst_map, index_class_dict):
    # Replace indices with classes in the instance map
    inst_map_replaced = np.zeros_like(inst_map)
    for index, class_ in index_class_dict.items():
        inst_map_replaced[inst_map == index] = class_

    return inst_map_replaced


if os.path.exists('inst_maps_replaced.npy'):
    # If the file exists, load it
    all_inst_maps_replaced = np.load('inst_maps_replaced.npy', allow_pickle=True)
    print("Replaced instance maps loaded from 'inst_maps_replaced.npy'")
else:
    # If the file does not exist, process the .mat files
    # Get list of .mat files
    mat_files = [file for file in os.listdir('lizard_labels/Lizard_Labels/Labels/') if file.endswith('.mat')]

    # Sort .mat files to maintain order
    mat_files.sort()

    # Initialize list to store replaced instance maps for all images
    all_inst_maps_replaced = []

    # Loop through .mat files
    for file in mat_files:
        # Load index-class pairs for the current file
        index_class_dict = load_index_class_pairs(file[:-4])  # Remove extension (.mat) from file name

        # Load the instance segmentation map
        label = sio.loadmat(f'lizard_labels/Lizard_Labels/Labels/{file}')

        # Get the instance map
        inst_map = label['inst_map']

        # Replace indices with classes
        inst_map_replaced = replace_indices_with_classes(inst_map, index_class_dict)

        all_inst_maps_replaced.append(inst_map_replaced)
        print(f"{file} appended.")

    # Convert list to numpy array
    all_inst_maps_replaced = np.array(all_inst_maps_replaced)

    # Save the array to a file
    np.save('inst_maps_replaced.npy', all_inst_maps_replaced)
    print("Replaced instance maps saved to 'inst_maps_replaced.npy'")
