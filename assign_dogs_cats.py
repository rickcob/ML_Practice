import os
import shutil

original_dataset_dir = r'E:\documentos\Conda\PetImages'
train_cats_dir = r'E:\documentos\Conda\cats_and_dogs_small\train\cats'
validation_cats_dir = \
    r'E:\documentos\Conda\cats_and_dogs_small\validation\cats'
test_cats_dir = r'E:\documentos\Conda\cats_and_dogs_small\test\cats'
train_dogs_dir = r'E:\documentos\Conda\cats_and_dogs_small\train\dogs'
validation_dogs_dir = \
    r'E:\documentos\Conda\cats_and_dogs_small\validation\dogs'
test_dogs_dir = r'E:\documentos\Conda\cats_and_dogs_small\test\dogs'
fnames = [r'cat\{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    # dst = os.path.join(train_cats_dir, fname)
    shutil.copy(src, train_cats_dir)
fnames = [r'cat\{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    # dst = os.path.join(validation_cats_dir, fname)
    shutil.copy(src, validation_cats_dir)
fnames = [r'cat\{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    # dst = os.path.join(test_cats_dir, fname)
    shutil.copy(src, test_cats_dir)
fnames = [r'dog\{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    # dst = os.path.join(train_dogs_dir, fname)
    shutil.copy(src, train_dogs_dir)
fnames = [r'dog\{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    # dst = os.path.join(validation_dogs_dir, fname)
    shutil.copy(src, validation_dogs_dir)
fnames = [r'dog\{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    # dst = os.path.join(test_dogs_dir, fname)
    shutil.copy(src, test_dogs_dir)
