import numpy as np
import tensorflow as tf
from tensorflow.python.keras import MobileNetV2
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D
from tensorflow.python.keras.utils import to_categorical

def load_data():
    X_resized = np.load('X_resized.npy')
    y = np.load('y.npy')
    target_names = np.load('target_names.npy', allow_pickle=True)
    y_categorical = to_categorical(y, len(target_names))
    return X_resized, y_categorical, target_names

def train_model():
    X_resized, y_categorical, target_names = load_data()

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Sequential([
        Conv2D(3, (3, 3), padding='same', input_shape=(224, 224, 1)),  # Convert grayscale to RGB
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(target_names), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_resized, y_categorical, epochs=10, validation_split=0.2)
    model.save('trained_model.h5')

if __name__ == "__main__":
    train_model()
