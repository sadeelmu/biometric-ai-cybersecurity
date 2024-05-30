import cv2
import numpy as np
from tensorflow import keras
from tensorflow.python.keras.models import load_model

def load_model_and_labels():
    model = load_model('trained_model.h5')
    target_names = np.load('target_names.npy', allow_pickle=True)
    class_labels = {i: name for i, name in enumerate(target_names)}
    return model, class_labels

def authenticate(image_path, model, class_labels):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=[0, -1]) / 255.0

    prediction = model.predict(image)
    predicted_class = class_labels[np.argmax(prediction)]

    return predicted_class

if __name__ == "__main__":
    model, class_labels = load_model_and_labels()
    test_image_path = 'path_to_test_image.jpg'
    authenticated_user = authenticate(test_image_path, model, class_labels)
    print(f'Authenticated as: {authenticated_user}')
