import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# ─── 1. LOAD DATA ────────────────────────────────────────────────────────────

columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
df = pd.read_csv('driving_log.csv', names=columns)

# Fix absolute paths from simulator → use just the filename inside local IMG/
def fix_path(p):
    return os.path.join('IMG', os.path.basename(str(p).strip()))

df['center'] = df['center'].apply(fix_path)
df['steering'] = df['steering'].astype(float)

print(f"Total samples: {len(df)}")

# ─── 2. VISUALIZE STEERING DISTRIBUTION (Section 4 - Reviewing Dataset) ──────

plt.figure(figsize=(8, 4))
plt.hist(df['steering'], bins=25, color='steelblue', edgecolor='black')
plt.title('Steering Angle Distribution (Raw Data)')
plt.xlabel('Steering Angle')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('steering_distribution.png')
plt.show()
print("Histogram saved to steering_distribution.png")

# ─── 3. TRAIN / VALIDATION SPLIT ─────────────────────────────────────────────

X = df['center'].values
y = df['steering'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}  |  Validation samples: {len(X_val)}")

# ─── 4. DATA AUGMENTATION (Section 4 - applied to training only) ─────────────

def augment_image(img, steering):
    """Apply random augmentations to a single image and its steering angle."""

    # Flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering

    # Brightness adjustment
    if random.random() < 0.5:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        factor = 0.5 + random.random()          # 0.5 – 1.5
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Panning (horizontal + vertical shift)
    if random.random() < 0.5:
        h, w = img.shape[:2]
        dx = int(random.uniform(-0.1 * w, 0.1 * w))
        dy = int(random.uniform(-0.1 * h, 0.1 * h))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, M, (w, h))
        steering += dx * 0.002            # compensate steering for horizontal pan

    # Zooming
    if random.random() < 0.5:
        h, w = img.shape[:2]
        factor = random.uniform(1.0, 1.3)
        new_h, new_w = int(h * factor), int(w * factor)
        img = cv2.resize(img, (new_w, new_h))
        # Centre-crop back to original size
        y0 = (new_h - h) // 2
        x0 = (new_w - w) // 2
        img = img[y0:y0 + h, x0:x0 + w]

    # Rotation (small angle only to stay realistic)
    if random.random() < 0.5:
        h, w = img.shape[:2]
        angle = random.uniform(-5, 5)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h))

    return img, steering

# ─── 5. PREPROCESSING (Section 5) ────────────────────────────────────────────

def preprocess(img):
    """Crop road area, convert to YUV, blur, resize, normalise."""
    img = img[60:135, :, :]                     # crop sky / hood  (Figure 6)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))            # Nvidia input size
    img = img / 255.0                           # normalise to [0, 1]
    return img

def load_and_preprocess(path, steering, augment=False):
    img = mpimg.imread(path)                    # reads as RGB
    if augment:
        img, steering = augment_image(img, steering)
    img = preprocess(img)
    return img, steering

# ─── 6. BATCH GENERATOR (Section 6) ──────────────────────────────────────────

def batch_generator(image_paths, steerings, batch_size=32, augment=False):
    """Yields (X_batch, y_batch) indefinitely, shuffling each epoch."""
    n = len(image_paths)
    indices = np.arange(n)
    while True:
        np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch, y_batch = [], []
            for i in batch_idx:
                img, steer = load_and_preprocess(
                    image_paths[i], steerings[i], augment=augment
                )
                X_batch.append(img)
                y_batch.append(steer)
            yield np.array(X_batch), np.array(y_batch)

# ─── 7. BUILD NVIDIA CNN MODEL (Section 7 / Figure 7) ────────────────────────

def build_model():
    model = Sequential([
        # Normalisation inside the network (input already /255 but Lambda keeps graph clean)
        Lambda(lambda x: x, input_shape=(66, 200, 3)),  # pass-through; data already normalised

        # 5 Convolutional layers
        Conv2D(24, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(36, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(48, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(64, (3, 3), activation='elu'),
        Conv2D(64, (3, 3), activation='elu'),

        Flatten(),

        # 3 Fully-connected layers
        Dense(100, activation='elu'),
        Dense(50,  activation='elu'),
        Dense(10,  activation='elu'),

        # Output: single steering angle
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
    return model

model = build_model()
model.summary()

# ─── 8. TRAIN ────────────────────────────────────────────────────────────────

BATCH_SIZE  = 32
EPOCHS      = 10
steps_train = len(X_train) // BATCH_SIZE
steps_val   = len(X_val)   // BATCH_SIZE

train_gen = batch_generator(X_train, y_train, batch_size=BATCH_SIZE, augment=True)
val_gen   = batch_generator(X_val,   y_val,   batch_size=BATCH_SIZE, augment=False)

checkpoint = ModelCheckpoint('model.h5', monitor='val_loss',
                             save_best_only=True, verbose=1)

history = model.fit(
    train_gen,
    steps_per_epoch=steps_train,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=steps_val,
    callbacks=[checkpoint],
    verbose=1,
)

# ─── 9. PLOT TRAINING CURVES ──────────────────────────────────────────────────

plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'],     label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training History')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()
print("Training graph saved to training_history.png")
print("Best model saved to model.h5")
