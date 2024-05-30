import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people

def preprocess_data():
    lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=0.4)
    X = lfw_people.images
    y = lfw_people.target
    target_names = lfw_people.target_names

    X_resized = np.array([cv2.resize(img, (224, 224)) for img in X])
    X_resized = X_resized / 255.0
    X_resized = np.expand_dims(X_resized, axis=-1)

    return X_resized, y, target_names

if __name__ == "__main__":
    X_resized, y, target_names = preprocess_data()
    np.save('X_resized.npy', X_resized)
    np.save('y.npy', y)
    np.save('target_names.npy', target_names)
