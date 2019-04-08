import cv2
from abc import abstractmethod
import numpy as np
import os
import sys
from ascii_graph import Pyasciigraph
import util


class DataManager:
    """
    This class provides train and test data for
    various tasks to a model in batch-sized portions.
    """
    def __init__(self, settings):
        self.batch_size = settings.get_setting_by_name('batch_size')

        # === GET TRAINING DATA ===
        if not os.path.exists(settings.get_setting_by_name('train_data_dir')):
            print('train data not found.')
            self.train_provider = OrderedBatchProvider([None], [None], 0)
        else:
            print('reading train data:')
            self.train_images, self.labels = self.get_labeled_data(settings, 'train')
            self.train_provider = RandomBatchProvider(self.train_images, self.labels, self.batch_size)

        # === GET TEST DATA ===
        if not os.path.exists(settings.get_setting_by_name('test_data_dir')):
            print('test data not found.')
            self.test_provider = OrderedBatchProvider([None], [None], 0)
        else:
            print('reading test data:')
            self.test_images, self.test_labels = self.get_labeled_data(settings, 'test')
            self.test_provider = OrderedBatchProvider(self.test_images, self.test_labels, self.batch_size)

    def next_batch(self):
        """
        Returns next train batch.
        :return: images and labels as numpy arrays. First dimension is equal to batch_size
        """
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

    def batches_per_test(self):
        """
        :return: the number of batches per epoch
        """
        return int(len(self.test_images)/self.batch_size)

    @abstractmethod
    def get_labeled_data(self, settings, stage):
        data = []
        labels = []
        return data, labels


class ClassificationDataManager(DataManager):
    def __init__(self, settings):
        super().__init__(settings)

    def get_labeled_data(self, settings, stage, print_distribution=True):
        """
        Reads all images from subdirectories and creates corresponding labels.
        Optionally, a sample distribution over the classes will be printed.
        Class names of generated labels will be saved in settings.
        :param settings: settings for providing data paths and for saving class names
        :param stage: decides which data is accessed depending on the stage (choices=['train', 'test'])
        :param print_distribution: if true, a distribution of samples over the classes will be printed to console.
        :return: the images and labels.
        """
        if stage not in ['train', 'test']:
            print('ERROR: invalid data manager stage!')
            sys.exit(0)

        path_to_data = settings.get_setting_by_name('train_data_dir') if stage == 'train' else settings.get_setting_by_name('test_data_dir')

        images = []
        labels = []

        distribution = []
        classes = []

        subdirs = list(os.walk(path_to_data))[0][1]
        subdirs = sorted(subdirs)
        for i in range(len(subdirs)):
            path = os.path.join(path_to_data, subdirs[i])
            list_dir = os.listdir(path)
            counter = 0
            for image in list_dir:
                if '.csv' in image:
                    continue
                image = cv2.imread(os.path.join(path, image))
                if len(np.shape(image)) < 3:
                    image = np.expand_dims(image, 2)
                images.append(util.transform(image, settings))
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
        images = np.resize(images, [len(images), settings.get_setting_by_name('height'),
                                    settings.get_setting_by_name('width'), settings.get_setting_by_name('channels')])
        labels = np.array(labels)

        if stage == 'train':
            settings.update({
                'class_names': classes,
                'num_classes': len(classes)
            })
        elif stage == 'test':
            settings.update({
                'class_names_test': classes
            })

        return images, labels


class AutoencodingDataManager(DataManager):
    def __init__(self, settings):
        super().__init__(settings)

    def get_labeled_data(self, settings, stage, image_limit=None):
        """
        reads all video/image files, transforms them into the correct shape
        and stacks them into a single array.
        :param: settings: used for retrieving the path to the data and the correct image dimensions
        :return All images, once as data, once as labels
        """
        if stage not in ['train', 'test']:
            print('ERROR: invalid data manager stage!')
            sys.exit(0)

        path_to_data = settings.get_setting_by_name('train_data_dir') if stage == 'train' else settings.get_setting_by_name('test_data_dir')
        images = util.read_any_data(path_to_data, settings=settings)
        print("\nTotal number of training images: " + str(len(images)))
        if image_limit is not None:
            images = np.random.permutation(images)
            images = images[0:image_limit]
            print("\nTotal number stochastically reduced to image limit: " + str(len(images)))

        images = np.resize(images, [len(images), settings.get_setting_by_name('height'),
                                    settings.get_setting_by_name('width'), settings.get_setting_by_name('channels')])
        return images, images


class SegmentationDataManager(DataManager):
    def __init__(self, settings):
        super().__init__(settings)

    def get_labeled_data(self, settings, stage):
        """
        reads all video/image files, transforms them into the correct shape
        and stacks them into a single array.
        :param: settings: used for retrieving the path to the data and the correct image dimensions
        :return All images, once as data, once as labels
        """

        if stage == 'train':
            path_to_data = settings.get_setting_by_name('train_data_dir')
            path_to_ground_truth = settings.get_setting_by_name('train_data_dir_gt')
        elif stage == 'test':
            path_to_data = settings.get_setting_by_name('test_data_dir')
            path_to_ground_truth = settings.get_setting_by_name('test_data_dir_gt')
        else:
            print('ERROR: invalid data manager stage!')
            sys.exit(0)

        images, image_names = util.read_any_data(path_to_data, settings=settings, return_filenames=True)
        ground_truth, ground_truth_names = util.read_any_data(path_to_ground_truth, settings=settings, return_filenames=True)
        if len(images) != len(ground_truth):
            print('ERROR: number of images and ground truth dont\'t match')
            sys.exit(0)

        for i in range(len(image_names)): # hopefully not necessary if names match
            if image_names[i] != ground_truth_names[i]:
                print('ERROR: names of images and ground truth don\'t match')
        print("\nTotal number of training examples: " + str(len(images)))

        images = np.resize(images, [len(images), settings.get_setting_by_name('height'),
                                    settings.get_setting_by_name('width'), settings.get_setting_by_name('channels')])
        # TODO: ground truth dims may differ from input dims
        ground_truth = np.resize(images, [len(ground_truth), settings.get_setting_by_name('height'),
                                          settings.get_setting_by_name('width'), settings.get_setting_by_name('channels')])
        return images, ground_truth


class BatchProvider:
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
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
        self.data, self.labels = self.shuffle_two_arrays_in_place(self.data, self.labels)

    @staticmethod
    def shuffle_two_arrays_in_place(a, b):
        """
        This function shuffles two arrays a and b the exact same way, without making a copy of either.
        This is important, because when training an autoencoder, a and b reference the same array.
        Copying an array and then shuffling instead of shuffling the arrays in place would therefore
        lead to inconsistent shuffles.
        :param a: First array to be shuffled
        :param b: Second array to be shuffled
        :return: The references to the identically shuffled arrays a and b
        (this works if a and b reference the same array)
        """
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        return a, b
