import tensorflow as tf

import numpy as np

import os

from .dataset_base import Dataset

import PIL

import gzip


class Fashion_MNIST(Dataset):

    def _reOrganizeDataset(self, base_dir):

        def saveImages(target_path, images, labels, batch_id):

            file_names = []

            num_data = len(labels)

            prog_bar = tf.keras.utils.Progbar(target=num_data)
            for k in range(num_data):

                img_name = os.path.join(target_path,
                                        'c%02d'%(labels[k] + 1),
                                        '%02d_%05d_%02d.png'%(batch_id + 1, k + 1, labels[k] + 1))

                with tf.device('CPU:0'):
                    if not tf.io.gfile.exists(img_name):
                        img = np.expand_dims(images[k], -1)
                        img_decoded = tf.image.encode_png(img)
                        with tf.io.gfile.GFile(img_name, 'bw') as f:
                            f.write(img_decoded.numpy())

                file_names.append(img_name)

                prog_bar.add(1)

            return file_names

        base = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
        files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
                 't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']

        paths = []

        print(self.information('Downloading and extracting the datasets...'))
        print(base_dir)
        for fname in files:
            if not os.path.exists(os.path.join(base_dir, fname)):
                tf.keras.utils.get_file(fname = fname,
                                        origin=base + fname,
                                        cache_dir=base_dir,
                                        extract=True)
                paths.append(os.path.join(base_dir, os.path.join('datasets',fname)))


        # original dataset organization
        with gzip.open(paths[0], 'rb') as lbpath:
            train_labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)

        with gzip.open(paths[1], 'rb') as imgpath:
            train_data = np.frombuffer(imgpath.read(), np.uint8,
                                    offset=16).reshape(len(train_labels), 28, 28)

        with gzip.open(paths[2], 'rb') as lbpath:
            test_labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)

        with gzip.open(paths[3], 'rb') as imgpath:
            test_data = np.frombuffer(imgpath.read(), np.uint8,
                                   offset=16).reshape(len(test_labels), 28, 28)

        num_classes = {'train': 10, 'val': 0, 'eval': 10}

        # create related folders for the reorganizaton (create datasets with raw images)
        path_to_train_images = os.path.join(base_dir, 'images', 'training')
        path_to_test_images = os.path.join(base_dir, 'images', 'evaluation')

        if not tf.io.gfile.exists(path_to_train_images):
            tf.io.gfile.makedirs(path_to_train_images)
            [tf.io.gfile.makedirs(os.path.join(path_to_train_images, 'c%02d'%(i + 1))) for i in range(num_classes['train'])]
            print(self.information('Created class folders for %s'%path_to_train_images))

        if not tf.io.gfile.exists(path_to_test_images):
            tf.io.gfile.makedirs(path_to_test_images)
            [tf.io.gfile.makedirs(os.path.join(path_to_test_images, 'c%02d'%(i + 1))) for i in range(num_classes['eval'])]
            print(self.information('Created class folders for %s' % path_to_test_images))


        # compute datasets mean to be used in preprocessing later
        dataset_mean = np.mean(train_data, axis=0).astype(np.float32) / 255

        # now write raw images
        example_names = {}

        print(self.information('Writing training images'))
        example_names['train'] = saveImages(path_to_train_images, train_data, train_labels, 0)

        example_names['val'] = None # saveImages(path_to_valid_images, valid_data, valid_labels, 1)

        print(self.information('Writing evaluation images'))
        example_names['eval'] = saveImages(path_to_test_images, test_data, test_labels, 2)

        shard_size = 10000

        return example_names, num_classes, shard_size, dataset_mean.tolist(), 1.


    def _exampleFeatures(self, example_name):
        # returns a list of arguments for tf examples extracted from the data of example_name

        image_path = example_name
        label = int(image_path.split('_')[-1].split('.')[0])

        image = PIL.Image.open(image_path, 'r')
        height = image.height
        width = image.width

        arg_list = [{'image_path': image_path, 'label': label - 1, 'height': height, 'width': width}]

        return arg_list

