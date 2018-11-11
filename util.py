import os
import sys
import csv
import zipfile

import Augmentor
import requests
from shutil import copyfile


def arrange_data_into_class_folders(src_dir, out_dir, path_to_annotations):
    """
    Create a directory for each class and move the images to the directory corresponding to their class.
    :param src_dir: directory containing the images.
    :param out_dir: root directory for class directories to be created.
    :param path_to_annotations: path to annotation file (assigning classes to images)
    """
    if not os.path.exists(src_dir):
        print('src dir does not exist!')
        sys.exit(0)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with open(path_to_annotations) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0

        class_index = -1
        name_index = -1
        for row in csv_reader:
            if line_count == 0:
                for i in range(len(row)):
                    print(row[i])
                    if 'class' in row[i] or 'Class' in row[i]:
                        class_index = i
                    if 'name' in row[i] or 'Name' in row[i]:
                        name_index = i
                if class_index == -1 or name_index == -1 or class_index == name_index:
                    print('need help determining name and class index/position :/')
                    sys.exit(0)
                line_count += 1
            else:
                file_name = row[name_index]
                class_number = row[class_index]
                path = out_dir + os.sep + class_number
                if not os.path.exists(path):
                    os.mkdir(path)
                copyfile(src_dir + os.sep + file_name, path + os.sep + file_name)
                line_count += 1


def get_necessary_data(dataset_name, data_dir):
    """
    If data set is missing, download und unzip data set from cloud.
    :param dataset_name: name of the data set.
                         Choices: 'isf' (Custom Image Set ISF Löwen) or 'gtsrb' (German Traffic Sign Recognition Benchmark)
    :param data_dir: the directory, to which the data set will be extracted
    :return: path to the training data directory and path to the test data directory
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(data_dir + os.sep + dataset_name):
        print('downloading..')
        if dataset_name == 'isf':
            download_file_from_google_drive('1Xvw7w3XKNLPWwfCZMKordtS7c-sIc_cs', data_dir + os.sep + 'data.zip')
        elif dataset_name == 'gtsrb':
            download_file_from_google_drive('1pD3PhvGhd8vXRazfzCjXe7dkNvXiUVtc', data_dir + os.sep + 'data.zip')
        print('unzipping..')
        zip_ref = zipfile.ZipFile(data_dir + os.sep + 'data.zip', 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()
        print('deleting zip..')
        os.remove(data_dir + os.sep + 'data.zip')
    return data_dir + os.sep + dataset_name + os.sep + 'train', data_dir + os.sep + dataset_name + os.sep + 'test'


def augment_data(scalar, data_dir, output_dir='auto', balance='False'):
    """
    Augments images of a given dataset via various operations,
    including: rotation, shearing, brightness and distortion or a any combination of those.
    :param scalar: scalar determining the ammount of generated images
                   based on the number of currently present images.
    :param data_dir: the directory including the folder for each class which in turn contain the images to be augmented
    :param output_dir: the directory to which the augmented images will be written.
                       The class structure via folders will be preserved.
                       If output_dir == 'auto', the augmented images will be added to the current structure.
    :param balance: whether the dataset should be balanced automatically, so that each class has the same amount of samples.
    """
    print('augmenting data..')
    distribution = []
    dirs = []
    subdirs = list(os.walk(data_dir))[0][1]
    for i in range(len(subdirs)):
        path = os.path.join(data_dir, subdirs[i])
        list_dir = os.listdir(path)
        counter = 0
        for image in list_dir:
            if '.csv' in image:
                os.remove(os.path.join(path, image))
            else:
                counter += 1

        distribution.append(counter)
        dirs.append(path)

    max_samples = max(distribution)

    for i in range(len(dirs)):
        path = dirs[i]
        pipeline = Augmentor.Pipeline(path, output_directory=(os.path.abspath(path) if output_dir == 'auto' else output_dir))
        pipeline.random_brightness(probability=0.5, min_factor=0.5, max_factor=1.5)
        pipeline.random_distortion(probability=0.1, grid_width=4, grid_height=4, magnitude=4)
        pipeline.rotate(probability=0.1, max_left_rotation=10, max_right_rotation=10)
        pipeline.shear(probability=0.1, max_shear_left=15, max_shear_right=15)

        if balance:
            pipeline.sample(int(max_samples * scalar - distribution[i]))
        else:
            pipeline.sample(int(distribution[i] * scalar))


def download_file_from_google_drive(id, destination):
    """
    Downloads file from google drive by id
    :param id: id of file to be downloaded.
    :param destination: destination where file will be saved to.
    """

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
