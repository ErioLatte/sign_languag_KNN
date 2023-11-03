# sign_languag_KNN
College_project

Tools used: 
Tensorflow, scikit-learn, 

Dataset source: 
kaggle bisindo datasets 
(https://www.kaggle.com/datasets/achmadnoer/alfabet-bisindo)

Paper:
https://docs.google.com/document/d/1pvA8mK5q3EEQRKWmCKGxgki3s9ACPSQ9qC67mACMpRM/edit?usp=sharing

This project is made to detect sign language using one of the popular ML model (KNN). By utilizing image augmentation and feature extraction using XCeption model, the KNN model acquired ~90% accuracy.

Some testing is done in the "playground" file.

Struggles:
1. the accuracy is bad (at first I used pixel per value for feature)
2. Not enough datasets

Solution
1. Use pretrained deep learning model to extract the feature.
2. Implemented image augmentation to increase dataset size

Speculation:
I dont think the model is read, because the good accuracy probably caused by image augmentation (which can lead to overfitting)