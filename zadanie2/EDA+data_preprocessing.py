import scipy.io as sio
from PIL import Image
import pandas as pd
from scipy.ndimage import zoom
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import exposure
from sklearn.utils import shuffle
from ipywidgets import interact, IntSlider


# def plot_images(idx):
#     plt.figure(figsize=(15, 6))
#     start_idx = idx * num_samples_per_row
#     end_idx = min(start_idx + num_samples_per_row, len(X_train))
#
#     for i in range(start_idx, end_idx):
#         # Plot the original image
#         plt.subplot(2, num_samples_per_row, i - start_idx + 1)
#         plt.imshow(X_train[i])
#         plt.title(f'Original Image {i}')
#         plt.axis('off')
#
#         # Plot the corresponding label
#         plt.subplot(2, num_samples_per_row, num_samples_per_row + i - start_idx + 1)
#         plt.imshow(y_train[i], cmap='viridis')
#         plt.title(f'Label {i}')
#         plt.axis('off')
#
#     plt.tight_layout()
#     plt.show()


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


def replace_indices_with_classes(inst_map, index_class_dict):
    # Replace indices with classes in the instance map
    inst_map_replaced = np.zeros_like(inst_map)
    for index, class_ in index_class_dict.items():
        inst_map_replaced[inst_map == index] = class_

    return inst_map_replaced


# Function to resize inst_maps to 500x500
def resize_inst_map(inst_map, new_shape=(500, 500)):
    return zoom(inst_map, (new_shape[0] / inst_map.shape[0], new_shape[1] / inst_map.shape[1]), order=0)


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


def load_index_class_pairs(pic):
    label = sio.loadmat(f'lizard_labels/Lizard_Labels/Labels/{pic}.mat')

    # Load the index array.
    nuclei_id = label['id'].squeeze().tolist()  # Convert to list

    # Load the nuclear categories / classes.
    classes = label['class'].squeeze().tolist()  # Convert to list

    # Combine index and class into a dictionary
    index_class_dict = {index: class_ for index, class_ in zip(nuclei_id, classes)}

    return index_class_dict


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


# Resize all_inst_maps_replaced to 500x500
resized_inst_maps = [resize_inst_map(inst_map) for inst_map in all_inst_maps_replaced]

# Convert list to numpy array
resized_inst_maps = np.array(resized_inst_maps)

# Save the array to a file
np.save('inst_maps_replaced_resized.npy', resized_inst_maps)
print("Replaced instance maps saved to 'inst_maps_replaced_resized.npy'")
print("Resized instance maps shape:", resized_inst_maps.shape)

# Augmentation
# Load data from .npy files
X = np.load('images_array.npy')  # Assuming X.npy contains your images
y = np.load('inst_maps_replaced_resized.npy')  # Assuming y.npy contains your labels

# Split the dataset into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Define data augmentation parameters
datagen = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=True)

# Fit the augmentation parameters on the training data
# datagen.fit(X_train)

# Generate augmented images and labels
augmented_images = []
augmented_labels = []

for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=len(X_train), shuffle=False):
    augmented_images.append(X_batch)
    augmented_labels.append(y_batch)
    break

# Convert augmented images and labels to numpy arrays
augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

# Reshape augmented images and labels to match the dimensions of X_train and y_train
augmented_images = augmented_images.reshape(-1, augmented_images.shape[2], augmented_images.shape[3], augmented_images.shape[4])
augmented_labels = augmented_labels.reshape(augmented_labels.shape[1], -1, augmented_labels.shape[2])

# # Concatenate augmented images and labels with original training data
# X_train = np.concatenate((X_train, augmented_images), axis=0)
#
# # Repeat y_train to match the number of augmented labels
# num_repeats = augmented_labels.shape[0] // y_train.shape[0]
# y_train = np.repeat(y_train, repeats=num_repeats, axis=0)
#
# # Concatenate augmented labels with original labels
# y_train = np.concatenate((y_train, augmented_labels), axis=0)

# Shuffle the training data
shuffle_index = np.random.permutation(len(X_train))
X_train = X_train[shuffle_index]
y_train = y_train[shuffle_index]

# Normalize the augmented images
mean_pixel_value = np.mean(X_train)
std_pixel_value = np.std(X_train)
normalized_images = (X_train - mean_pixel_value) / std_pixel_value

# Optionally, apply contrast enhancement (e.g., histogram equalization) to the normalized images
enhanced_images = []
for image in normalized_images:
    enhanced_channels = []
    for channel in range(image.shape[-1]):  # Iterate over each color channel
        enhanced_channel = exposure.equalize_hist(image[:, :, channel])
        enhanced_channels.append(enhanced_channel)
    enhanced_image = np.stack(enhanced_channels, axis=-1)  # Stack the enhanced channels back together
    enhanced_images.append(enhanced_image)
enhanced_images = np.array(enhanced_images)

# Concatenate original and enhanced images
X_train_augmented = np.concatenate((X_train, enhanced_images), axis=0)
# Update labels for enhanced images
y_train_augmented = np.concatenate((y_train, y_train), axis=0)  # Assuming labels are in the same order as images
# Shuffle the augmented dataset
X_train_augmented, y_train_augmented = shuffle(X_train_augmented, y_train_augmented)

X_train_augmented = X_train_augmented.astype(np.float32) / 255.0
X_train = X_train.astype(np.float32) / 255.0

# # Define the number of samples to visualize and number of samples per row
# num_samples = 284
# num_samples_per_row = 10  # Adjust this value as needed
#
# # Create an interactive plot using ipywidgets
# interact(plot_images, idx=IntSlider(min=0, max=(len(X_train) - 1) // num_samples_per_row, step=1, value=0))

# Define the number of samples to visualize
num_samples = 10

# Select random indices from the dataset
indices = np.random.choice(len(X_train), num_samples, replace=False)

# Plot the original images and their corresponding labels
plt.figure(figsize=(15, 6))
for i, idx in enumerate(indices):
    # Plot the original image
    plt.subplot(2, num_samples, i + 1)
    # you can use X_train, or enhanced_images here below (enhanced images are normalized in colour)
    plt.imshow(enhanced_images[idx])
    plt.title(f'Original Image {idx}')
    plt.axis('off')

    # Plot the corresponding label
    plt.subplot(2, num_samples, num_samples + i + 1)
    plt.imshow(y_train[idx], cmap='viridis')
    plt.title(f'Label {idx}')
    plt.axis('off')

plt.tight_layout()
plt.show()
