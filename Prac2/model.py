import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import shutil
import random

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Set paths
image_folder = '/content/drive/MyDrive/Med/data/training_set/images'
df = pd.read_csv(r'/content/drive/MyDrive/Med/data/train.csv')

# Image settings and normalization factor
IMG_SIZE = 224
MAX_HC = df['head circumference (mm)'].max()

def load_and_resize_image(filename, img_folder):
    img_path = os.path.join(img_folder, filename)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img

def augment_image(image):
    augmented_images = []
    augmented_images.append(image)
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)
    angle = random.uniform(-10, 10)
    h, w = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    augmented_images.append(rotated)

    return augmented_images

# Split into train (600) and temp (400)
train_df, temp_df = train_test_split(
    df,
    train_size=600,
    random_state=42,
    shuffle=True
)

# Split temp into val (200) and test (200)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    shuffle=True
)

augmented_images = []
augmented_hc_values = []

for idx, row in train_df.iterrows():
    filename = row['filename']
    hc_mm = row['head circumference (mm)']

    img = load_and_resize_image(filename, image_folder)

    augmented_versions = augment_image(img)

    for aug_img in augmented_versions:
        augmented_images.append(aug_img)
        augmented_hc_values.append(hc_mm)

X_train = np.array(augmented_images, dtype=np.float32)
y_train = np.array(augmented_hc_values, dtype=np.float32) / MAX_HC

# Validation Set 

val_images = []
val_hc_values = []

for idx, row in val_df.iterrows():
    filename = row['filename']
    hc_mm = row['head circumference (mm)']

    img = load_and_resize_image(filename, image_folder)

    val_images.append(img)
    val_hc_values.append(hc_mm)

X_val = np.array(val_images, dtype=np.float32)
y_val = np.array(val_hc_values, dtype=np.float32) / MAX_HC

# Prepare Test Set 

test_images = []
test_hc_values = []

for idx, row in test_df.iterrows():
    filename = row['filename']
    hc_mm = row['head circumference (mm)']

    img = load_and_resize_image(filename, image_folder)

    test_images.append(img)
    test_hc_values.append(hc_mm)

X_test = np.array(test_images, dtype=np.float32)
y_test = np.array(test_hc_values, dtype=np.float32) / MAX_HC

input_shape = (224, 224, 3)

# Load EfficientNetB2
base_model = EfficientNetB2(include_top=False, weights='imagenet', input_shape=input_shape)
base_model.trainable = True

inputs = tf.keras.Input(shape=input_shape)
x = base_model(inputs, training=True)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.7)(x)  # High dropout to prevent overfitting
outputs = layers.Dense(1, activation='linear')(x)

model = models.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='mae',
    metrics=['mae']
)

checkpoint = ModelCheckpoint(
    'best_model.h5',             
    monitor='val_loss',          
    save_best_only=True,         
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',         
    patience=10,                
    restore_best_weights=True,   
    verbose=1
)

callbacks = [checkpoint, early_stop]

epochs = 100
batch_size = 16

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks
)
