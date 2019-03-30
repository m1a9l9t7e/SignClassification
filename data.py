import cv2
from abc import abstractmethod
import numpy as np
import os
import sys
from ascii_graph import Pyasciigraph
from data_generator import Generator


class DataManager:
    """
    This class reads and augments the data and ultimately provides
    train and test data to a model in batch-sized portions.
    """
    def __init__(self, settings):
        self.height = settings.get_setting_by_name('height')
        self.width = settings.get_setting_by_name('width')
        self.channels = settings.get_setting_by_name('channels')
        self.batch_size = settings.get_setting_by_name('batch_size')

        if not os.path.exists(settings.get_setting_by_name('train_data_dir')):
            print('train data not found.')
            classes_train = [None]
            self.train_provider = OrderedBatchProvider([None], [None], 0, classes_train)
        else:
            print('reading train data:')
            self.train_images, self.labels, classes_train = self.read(settings.get_setting_by_name('train_data_dir'))
            self.train_provider = RandomBatchProvider(self.train_images, self.labels, self.batch_size, classes_train)

        if not os.path.exists(settings.get_setting_by_name('test_data_dir')):
            print('test data not found.')
            classes_test = [None]
            self.test_provider = OrderedBatchProvider([None], [None], 0, classes_test)
        else:
            print('reading test data:')
            self.test_images, self.test_labels, classes_test = self.read(settings.get_setting_by_name('test_data_dir'))
            self.test_provider = OrderedBatchProvider(self.test_images, self.test_labels, self.batch_size, classes_test)

        if not (len(classes_train) <= 1 or len(classes_test) <= 1):  # either test or train data was not found
            if len(classes_train) != len(classes_test):
                print("Warning: number of classes of train and test set don't match!")
                # sys.exit(0)
            else:
                for i in range(len(classes_train)):
                    if classes_train[i] != classes_test[i]:
                        print('ERROR: train and test classes don\'t match.')
                        sys.exit(0)
            settings.update({'num_classes': len(classes_train),
                             'class_names': classes_train,
                             'class_names_test': classes_test})

        if settings.get_setting_by_name('use_synthetic_training_data'):
            self.generator = Generator(settings, random_colors=False, path_to_background=settings.get_setting_by_name('path_to_background_data'),
                                       path_to_foreground=settings.get_setting_by_name('path_to_foreground_data'))
            generator_class_names = self.generator.get_class_names()
            settings.update({'class_names': generator_class_names})
            settings.update({'num_classes': len(generator_class_names)})
            if len(generator_class_names) != len(classes_train):
                print('Warning: classes in training set and synthetic data don\'t match! This will lead to inconsistent class labels.')
                print('You should also make sure that class names match!')
                print('Aborting.')
                # TODO: make distinction between mixed training and pure synthetic. In case of mixed, stop trainign if
                # TODO: artif and trainset data classes don't match
                # sys.exit(0)
        else:
            self.generator = None

    def image_conversion(self, image):
        """
        Converts image to grayscale and resizes according to settings.
        :param image: the image to be converted.
        :return: the converted image as numpy array.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, ) if self.channels == 1 else image
        image = cv2.resize(image, (self.height, self.width))
        return image

    def read(self, data_dir, print_distribution=True):
        """
        Reads all images from subdirectories and creates corresponding labels.
        Optionally, a sample distribution over the classes will be printed.
        :param data_dir: the directory containing a folder for each class which in turn contain the data.
        :param print_distribution: if true, a distribution of samples over the classes will be printed to console.
        :param channels: number of channels for each image (has to be smaller or equal to channels of raw data)
        :return: the images, labels and number of classes.
        """
        images = []
        labels = []

        distribution = []
        classes = []

        subdirs = list(os.walk(data_dir))[0][1]
        subdirs = sorted(subdirs)
        for i in range(len(subdirs)):
            path = os.path.join(data_dir, subdirs[i])
            list_dir = os.listdir(path)
            counter = 0
            for image in list_dir:
                if '.csv' in image:
                    continue
                image = cv2.imread(os.path.join(path, image))
                if len(np.shape(image)) < 3:
                    image = np.expand_dims(image, 2)
                images.append(self.image_conversion(image))
                label = np.zeros(len(subdirs), np.float32)
                label[i] = 1.0
                labels.append(label)
                counter += 1
            sys.stdout.write('\rreading files from subdirectory ' + str(i+1) + '/' + str(len(subdirs)))
            sys.stdout.flush()
            classes.append(subdirs[i])
            distribution.append((subdirs[i], counter))

        if print_distribution:
            graph = Pyasciigraph()
            for line in graph.graph('\nclass distribution:', distribution):
                print(line)
        print("total number of images: " + str(len(images)))
        print(np.shape(images))
        images = np.array(images)
        images = np.resize(images, [len(images), self.height, self.width, self.channels])
        labels = np.array(labels)

        return images, labels, classes

    def next_batch(self):
        """
        Returns next train batch.
        :return: images and labels as numpy arrays. First dimension is equal to batch_size
        """
        if self.generator is not None:
            return self.generator.generate(self.batch_size)
        else:
            return self.train_provider.next_batch()

    def next_test_batch(self):
        """
        Returns next test batch.
        :return: images and labels as numpy arrays. First dimension is equal to batch_size
        """
        return self.test_provider.next_batch()

    def batches_per_epoch(self):
        """
        :return: the number of batches per epoch
        """
        return int(len(self.train_images)/self.batch_size)


class BatchProvider:
    def __init__(self, data, labels, batch_size, class_names):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.class_names = class_names
        self.iteration = 0
        self.reset()

    def next_batch(self):
        """
        Returns next batch of size batch_size until no more data is available.
        If less data than a single batch_size remains, a single smaller batch will be returned.
        After this an empty batch will be returned to signal the end of the epoch and the dataset will be reset via the reset() function.
        :return: The next batch as specified above.
        """
        if self.iteration == -1:
            self.reset()
            return [], []
        if (self.iteration + 1) * self.batch_size > len(self.data) - 1:
            batch_x = self.data[self.iteration * self.batch_size:]
            batch_y = self.labels[self.iteration * self.batch_size:]
            self.iteration = -1
            return batch_x, batch_y
        else:
            batch_x = self.data[self.iteration * self.batch_size:(self.iteration + 1) * self.batch_size]
            batch_y = self.labels[self.iteration * self.batch_size:(self.iteration + 1) * self.batch_size]
            self.iteration += 1
            return batch_x, batch_y

    @abstractmethod
    def reset(self):
        self.iteration = 0


class OrderedBatchProvider(BatchProvider):
    """
    Provides data and labels in the order they were first read.
    This is suitable for testing and validation, because order is irrelevant in these cases.
    """
    def reset(self):
        self.iteration = 0


class RandomBatchProvider(BatchProvider):
    """
    Provides data and labels in a random order.
    This provider should be used for the train data to keep the model balanced.
    """
    def reset(self):
        """
        Resets the data by shuffling data and labels jointly
        :return:
        """
        self.iteration = 0
        perm = np.arange(len(self.data))
        np.random.shuffle(perm)
        self.data = self.data[perm]
        self.labels = self.labels[perm]
