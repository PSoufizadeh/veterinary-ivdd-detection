# Import necessary libraries
import os
import shutil
import random
import tensorflow as tf
from keras.applications import VGG16
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.models import Model, Sequential
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import classification_report

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Paths to the original dataset directories
original_dataset_dir_healthy = 'Healthy'
original_dataset_dir_ivdd = 'IVDD'

# Ensure that the class directories are named to enforce correct label assignment
if os.path.exists('IVDD'):
    os.rename('IVDD', '0_IVDD')
if os.path.exists('Healthy'):
    os.rename('Healthy', '1_Healthy')

# Update the paths after renaming
original_dataset_dir_healthy = '1_Healthy'
original_dataset_dir_ivdd = '0_IVDD'

# Define the base directory where we'll put our dataset
base_dir = 'dataset'
os.makedirs(base_dir, exist_ok=True)

# Define directories for train, validation, test splits
train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)
validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(validation_dir, exist_ok=True)
test_dir = os.path.join(base_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

# Create subdirectories for each class in each split
for split in ['train', 'validation', 'test']:
    for category in ['0_IVDD', '1_Healthy']:
        dir_path = os.path.join(base_dir, split, category)
        os.makedirs(dir_path, exist_ok=True)

# Function to split data
def split_data(original_dir, train_dir, validation_dir, test_dir, split_ratio=(0.7, 0.15, 0.15)):
    files = os.listdir(original_dir)
    files = [f for f in files if os.path.isfile(os.path.join(original_dir, f))]
    random.shuffle(files)
    total_files = len(files)
    train_split = int(split_ratio[0] * total_files)
    val_split = int(split_ratio[1] * total_files)
    train_files = files[:train_split]
    val_files = files[train_split:train_split + val_split]
    test_files = files[train_split + val_split:]

    # Copy files to respective directories
    for file in train_files:
        src = os.path.join(original_dir, file)
        dst = os.path.join(train_dir, file)
        shutil.copyfile(src, dst)
    for file in val_files:
        src = os.path.join(original_dir, file)
        dst = os.path.join(validation_dir, file)
        shutil.copyfile(src, dst)
    for file in test_files:
        src = os.path.join(original_dir, file)
        dst = os.path.join(test_dir, file)
        shutil.copyfile(src, dst)

# Split the Healthy and IVDD images
split_data(original_dataset_dir_healthy, os.path.join(train_dir, '1_Healthy'),
           os.path.join(validation_dir, '1_Healthy'), os.path.join(test_dir, '1_Healthy'))
split_data(original_dataset_dir_ivdd, os.path.join(train_dir, '0_IVDD'), os.path.join(validation_dir, '0_IVDD'),
           os.path.join(test_dir, '0_IVDD'))

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Data generators
batch_size = 16
target_size = (224, 224)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary',
    classes=['0_IVDD', '1_Healthy'],
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary',
    classes=['0_IVDD', '1_Healthy'],
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary',
    classes=['0_IVDD', '1_Healthy'],
    shuffle=False
)

# Generate synthetic masks using K-means clustering for segmentation
for split in ['train', 'validation']:
    for category in ['0_IVDD', '1_Healthy']:
        mask_category_dir = os.path.join(base_dir, f'{split}_masks', category)
        os.makedirs(mask_category_dir, exist_ok=True)
        image_category_dir = os.path.join(base_dir, split, category)
        for img_name in os.listdir(image_category_dir):
            img_path = os.path.join(image_category_dir, img_name)
            if os.path.isfile(img_path):
                # Read the image
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Enhance image contrast using CLAHE
                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                enhanced_img = cv2.merge((cl, a, b))
                enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2RGB)

                # Reshape the image for K-means clustering
                pixel_values = enhanced_img.reshape((-1, 3))
                pixel_values = np.float32(pixel_values)

                # Define criteria for K-means clustering
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                k = 3  # Number of clusters
                _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

                # Convert centers to 8-bit values
                centers = np.uint8(centers)

                # Map labels to pixel values
                segmented_image = centers[labels.flatten()]

                # Reshape back to the original image shape
                segmented_image = segmented_image.reshape(enhanced_img.shape)

                # Convert the segmented image to grayscale for mask
                mask = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)

                # Apply thresholding to create a binary mask
                _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

                # Save the generated mask
                mask_path = os.path.join(mask_category_dir, img_name)
                cv2.imwrite(mask_path, binary_mask)

# Modify data generators to fit U-Net model for segmentation (using images and masks)
train_image_datagen = ImageDataGenerator(rescale=1. / 255)
train_mask_datagen = ImageDataGenerator(rescale=1. / 255)

train_image_generator = train_image_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=True,
    seed=42
)

train_mask_generator = train_mask_datagen.flow_from_directory(
    os.path.join(base_dir, 'train_masks'),
    target_size=target_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=True,
    seed=42
)

# Combine the image and mask generators
train_generator_segmentation = zip(train_image_generator, train_mask_generator)

# Build U-Net model for disc localization
def build_unet(input_shape=(224, 224, 3)):
    inputs = tf.keras.layers.Input(input_shape)

    # Encoding
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

    # Decoding
    u1 = UpSampling2D((2, 2))(c3)
    u1 = concatenate([u1, c2])
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    u2 = UpSampling2D((2, 2))(c4)
    u2 = concatenate([u2, c1])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = Model(inputs, outputs)
    return model

unet_model = build_unet()
unet_model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
unet_model.summary()

# Train U-Net for segmentation
unet_model.fit(
    train_generator_segmentation,
    steps_per_epoch=len(train_image_generator),
    epochs=20,
    callbacks=[EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)]
)

# Build VGG16-based classifier with fine-tuning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Unfreeze the last block for fine-tuning
for layer in base_model.layers[:-4]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# Compile the model with lower learning rate
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the VGG16-based classifier with fine-tuning
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
               ModelCheckpoint('best_vgg16_model.h5', save_best_only=True),
               ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)]
)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Generate classification report
test_generator.reset()
predictions = model.predict(test_generator)
predicted_classes = np.where(predictions > 0.5, 1, 0).flatten()
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)


# Save images based on prediction correctness
true_dir = 'test_results/true'
false_dir = 'test_results/false'
os.makedirs(true_dir, exist_ok=True)
os.makedirs(false_dir, exist_ok=True)

predicted_classes = model.predict(test_generator).ravel() > 0.5
true_classes = test_generator.classes



for i in range(len(true_classes)):
    img_path = test_generator.filepaths[i]
    img_name = os.path.basename(img_path)
    predicted_label = '1_Healthy' if predicted_classes[i] == 1 else '0_IVDD'
    true_label = '1_Healthy' if true_classes[i] == 1 else '0_IVDD'
    if predicted_classes[i] == true_classes[i]:
        dst = os.path.join(true_dir, f"{true_label}_{img_name}")
    else:
        dst = os.path.join(false_dir, f"{predicted_label}_{img_name}")
    shutil.copyfile(img_path, dst)
