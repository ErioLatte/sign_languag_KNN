import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image


# path to dataset 
dataset_path = "./bisindo_dataset/"

# use imagedatagenerator to create augment data
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

# Load original images
for subpath in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, subpath)

    # this will iterate all the image file on each class folder
    for filename in os.listdir(class_path):
        img_path = os.path.join(class_path, filename)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = x.reshape((1,) + x.shape)  # Reshape to (1, height, width, channels)

        # Generate 35 augmented images for each image and save them
        i = 0
        for batch in datagen.flow(x, save_to_dir=class_path, save_prefix='augm_', save_format='jpg'):
            i += 1
            if i >= 35: 
                break
