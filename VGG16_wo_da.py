from keras.applications import VGG16
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150,
                                                                      3))
train_dir = '/home/richard/Documents/cats_and_dogs_small/train'
validation_dir = '/home/richard/Documents/cats_and_dogs_small/validation'
test_dir = '/home/richard/Documents/cats_and_dogs_small/test'
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20


def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(directory, target_size=(150, 150),
                                            batch_size=batch_size,
                                            class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        print(i)
        if i * batch_size >= sample_count:
            break
    return features, labels


train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)
