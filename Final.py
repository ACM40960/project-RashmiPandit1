import tensorflow as tf
import numpy as np
#import Detection
import os
import matplotlib.pyplot as m_plt

def detection(data_model, class_labels_map, type, photo_id):
    photo_path = "./" + type +"/" + type + "_" + str(photo_id) + ".JPG"
    m_plt.figure()
    photo_mtx = m_plt.imread(photo_path)
    m_plt.imshow(photo_mtx)
    m_plt.axis("Off")

    pic = tf.keras.preprocessing.image.load_img(photo_path, target_size=(112, 112))
    pic_arr = tf.keras.preprocessing.image.img_to_array(pic)
    temp = np.expand_dims(pic_arr, axis = 0)

    pic_preprocessed = np.vstack([temp])

    detections = data_model.predict(pic_preprocessed)
    print("\nProbabilities : ", detections)

    probability_max = np.argmax(detections)
    print("\nPredicted : ", list(class_labels_map.keys())[list(class_labels_map.values()).index(probability_max)])

    m_plt.show(block=False)
    
type1 = "chihuahua"
type2 = "muffin"

path_id = 0
path_name = ""
photo_id = 0

main_path = type1 + '_' + type2
train_path = os.path.join(main_path, "train")
validation_path = os.path.join(main_path, "validation")
test_path = os.path.join(main_path, "test")

# Loading datasets into tensors
train_dataset = tf.keras.utils.image_dataset_from_directory(
    directory = train_path,
    image_size = (112, 112),
    batch_size = 10)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    directory = validation_path,
    image_size = (112, 112),
    batch_size = 10)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    directory = test_path,
    image_size = (112, 112),
    batch_size = 10)

# Data augmentation generation
train_data_generation = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255,
    height_shift_range = 0.2,
    horizontal_flip = True,
    rotation_range = 40,
    width_shift_range = 0.2,
    zoom_range = 0.2,
    shear_range = 0.2
    )

validate_data_generation = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

train_generator = train_data_generation.flow_from_directory(
    train_path,
    target_size = (112, 112),
    batch_size = 10,
    class_mode = "binary")

validation_generator = validate_data_generation.flow_from_directory(
    validation_path,
    target_size = (112, 112),
    batch_size = 10,
    class_mode = "binary")

class_labels_map = train_generator.class_indices
cnn_model = tf.keras.models.load_model("vgg19_"+ type1+ "_" + type2 + ".h5")

while path_id != 3:
    print("Choose Selector: ")
    print("1. " + type1)
    print("2. " + type2)
    print("3. Exit\n")
    path_id = int(input())

    if path_id == 1:
        path_name = type1
        print("Valid id's for " + type1 + " 1-900")
        photo_id = int(input())
        if 1 <= photo_id <= 900:
            detection(cnn_model, class_labels_map, path_name, photo_id)
        else:
            print("Choose a valid photo id!")
    elif path_id == 2:
        path_name = type2
        print("Valid id's for "+ type2 + " 1-500")
        photo_id = int(input())
        if 1 <= photo_id <= 500:
            detection(cnn_model, class_labels_map, path_name, photo_id)
        else:
            print("Choose a valid instance id!")
    elif path_id == 3:
        break
    else:
        print("Choose a valid option!")