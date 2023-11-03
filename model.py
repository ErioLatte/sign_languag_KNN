import os
import numpy as np
from PIL import Image
from tensorflow import keras
from keras.applications import Xception
from keras.applications.xception import preprocess_input
from keras.preprocessing import image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# define path, get dataset path & label
dataset_path = "./bisindo_dataset/"
def get_dataset_dir(dataset_path):
    labels = [] # labels is used as Y
    img_dir = [] # contain all image directory

    # this will take image dir and its label
    for x in os.listdir(dataset_path):
        subpath = os.path.join(dataset_path, x)
        for y in os.listdir(subpath):
            labels.append(x)
            img_dir.append(os.path.join(subpath, y))
    return labels, img_dir

# from image_dir get image features using Xception model
def get_feature(path):
    model = Xception(weights="imagenet", include_top=False)
    features = []
    for x in path:
        img = image.load_img(x, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features.append(
            model.predict(x).flatten()
        )

    return features

# preprocess the datasets (train-test split, normalization)
def preprocess(features, labels):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Feature normalization
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test

# create the model, train, and predict the test dataset
def get_train_model(x_train, x_test, y_train, y_test, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)   

    # Predict image
    y_pred = knn.predict(x_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"- K:{k} | Accuracy: {accuracy}")

    return knn

## call the function above
labels, image_dir = get_dataset_dir(dataset_path)
image_features = get_feature(image_dir)
train_image, test_image, train_label, test_label = preprocess(image_features, labels)
model = get_train_model(train_image, test_image, train_label, test_label, 3)