import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import os

# Configuration
DATA_DIR = './project6'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
MODEL_PATH = './models/final_model.keras'

def train_model():
    print("Initializing MobileNetV2 for Drone vs. Bird Classification...")
    
    # Load dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    # Preprocessing: Rescale MobileNetV2 input
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)

    # Data Augmentation Layer
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])

    # Base Model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    # Model Structure
    model = models.Sequential([
        layers.Rescaling(1./255), # Scaling to [0, 1] as per UI requirements
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("Starting training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    print(f"Saving model to {MODEL_PATH}...")
    if not os.path.exists('./models'):
        os.makedirs('./models')
    model.save(MODEL_PATH)
    print("Model saved successfully.")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory {DATA_DIR} not found.")
    else:
        train_model()
