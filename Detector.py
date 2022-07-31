import os
import matplotlib.pyplot as m_plt
import tensorflow as tf
import numpy as np
import Detection

type1 = "chihuahua"
type2 = "muffin"

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

# Setup network
convolution_base = tf.keras.applications.VGG19(
    input_shape = (112, 112, 3), 
    include_top = False, 
    weights = "imagenet"
)

temp = convolution_base.output
temp = tf.keras.layers.Dense(128)(temp)
temp = tf.keras.layers.GlobalAveragePooling2D()(temp)
temp = tf.keras.layers.Dropout(0.3)(temp)
predictions = tf.keras.layers.Dense(2, activation="softmax")(temp)

cnn_model = tf.keras.models.Model(inputs = convolution_base.input, outputs = predictions)
convolution_base.trainable = False

cnn_model.compile(
    optimizer = tf.keras.optimizers.Adagrad(),
    loss = "sparse_categorical_crossentropy",
    metrics = ['acc']
)

n_training_pics = 250
n_validation_pics = 150
batch_size = 100

n_steps_epoch = n_training_pics / batch_size
n_validation_steps = n_validation_pics / batch_size

history = cnn_model.fit(
    train_generator,
    epochs = 20,
    validation_steps = n_validation_steps,
    steps_per_epoch = n_steps_epoch,
    validation_data = validation_generator
    )

cnn_model.save("vgg19_"+ type1+ "_" + type2 + ".h5")

training_accuracy = history.history['acc']
validation_accuracy = history.history['val_acc']
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs = range(1, len(training_accuracy) + 1)
m_plt.plot(epochs, training_accuracy, 'b', label = 'Train accuracy')
m_plt.plot(epochs, validation_accuracy, 'blue', label = 'Validation accuracy')
m_plt.title('Training accuracy vs Validation accuracy')
m_plt.legend()

m_plt.figure()

m_plt.plot(epochs, training_loss, 'b', label = 'Training loss')
m_plt.plot(epochs, validation_loss, 'blue', label = 'Validation loss')
m_plt.title('Training loss vs Validation loss')
m_plt.legend()

m_plt.show(block=False)

test_data_generation = tf.keras.preprocessing.image.ImageDataGenerator(1./255)
test_generator = test_data_generation.flow_from_directory(
    test_path,
    batch_size = 10,
    target_size = (112, 112),
    class_mode = "binary")

test_loss, test_accuracy = cnn_model.evaluate(test_generator, steps = n_steps_epoch)
print("\nTest Accuracy :", test_accuracy)

class_labels_map = train_generator.class_indices
print("\nClasses : ", class_labels_map)

'''
path_id = 0
path_name = ""
photo_id = 0

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
            Detection.detection(cnn_model, class_labels_map, path_name, photo_id)
        else:
            print("Choose a valid photo id!")
    elif path_id == 2:
        path_name = type2
        print("Valid id's for "+ type2 + " 1-500")
        photo_id = int(input())
        if 1 <= photo_id <= 500:
            Detection.detection(cnn_model, class_labels_map, path_name, photo_id)
        else:
            print("Choose a valid instance id!")
    elif path_id == 3:
        break
    else:
        print("Choose a valid option!")
'''
m_plt.show()
