import os
import shutil
import sys
import csv
import zipfile

import Augmentor
from PIL import Image
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
        print('ERROR: src dir does not exist!')
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
                    print('ERROR: need help determining name and class position')
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
        print('Downloading..')
        if dataset_name == 'isf':
            download_file_from_google_drive('1Xvw7w3XKNLPWwfCZMKordtS7c-sIc_cs', data_dir + os.sep + 'data.zip')
        elif dataset_name == 'gtsrb':
            download_file_from_google_drive('1SnQphh6TpDShavXDT6fpeyT2I9xzEa6T', data_dir + os.sep + 'data.zip')
        elif dataset_name == 'mnist':
            download_file_from_google_drive('1I5J1OZPti20w8zFGYsBqbyaiN7X8z4iB', data_dir + os.sep + 'data.zip')
        print('Unzipping..')
        zip_ref = zipfile.ZipFile(data_dir + os.sep + 'data.zip', 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()
        print('Deleting zip..')
        os.remove(data_dir + os.sep + 'data.zip')
    return data_dir + os.sep + dataset_name + os.sep + 'train', data_dir + os.sep + dataset_name + os.sep + 'test'


def augment_data(scalar, path_to_data, path_to_index, output_dir='auto', balance='False'):
    """
    Augments images of a dataset (given by index file) via various operations,
    including: rotation, shearing, brightness and distortion or a any combination of those.
    :param scalar: scalar determining the ammount of generated images
                   based on the number of currently present images.
    :param path_to_data: the directory including the folder for each class which in turn contain the images to be augmented
    :param output_dir: the directory to which the augmented images will be written.
                       The class structure via folders will be preserved.
                       If output_dir == 'auto', the augmented images will be added to the current structure.
    :param balance: whether the dataset should be balanced automatically, so that each class has the same amount of samples.
    """
    print('Augmenting data..')
    # move original images to temporary folder with help of index file
    file = open(path_to_index, 'r')
    paths_to_orig_images = file.readlines()
    path_to_moved_data = '.'+os.sep+'temp'
    moved_paths = move_files(path_to_data, path_to_moved_data, paths_to_orig_images, root=os.path.abspath(path_to_data))

    # count original images
    moved_subdirs = list(os.walk(path_to_moved_data))[0][1]
    original_distribution = []
    moved_dirs = []
    for i in range(len(moved_subdirs)):
        moved_path = os.path.join(path_to_moved_data, moved_subdirs[i])
        list_dir = os.listdir(moved_path)
        original_distribution.append(len(list_dir))
        moved_dirs.append(moved_path)

    # count artificially added images
    subdirs = list(os.walk(path_to_data))[0][1]
    artificial_distribution = []
    dirs = []
    for i in range(len(moved_subdirs)):
        path = os.path.join(path_to_data, subdirs[i])
        list_dir = os.listdir(path)
        artificial_distribution.append(len(list_dir))
        dirs.append(path)

    max_samples = max(original_distribution)
    min_samples = min(original_distribution)
    samples_counter = 0

    for i in range(len(dirs)):
        moved_path = moved_dirs[i]
        path = dirs[i]

        if balance:
            if min_samples * scalar < max_samples:
                samples = max_samples - artificial_distribution[i] - original_distribution[i]
            else:
                samples = min_samples * scalar - artificial_distribution[i] - original_distribution[i]
        else:
            samples = original_distribution[i] * scalar - original_distribution[i] - artificial_distribution[i]

        if samples > 0:
            # setup pipeline
            pipeline = Augmentor.Pipeline(os.path.abspath(moved_path), output_directory=(os.path.abspath(path) if output_dir == 'auto' else output_dir))
            pipeline.random_brightness(probability=0.5, min_factor=0.5, max_factor=1.5)
            pipeline.random_distortion(probability=0.1, grid_width=4, grid_height=4, magnitude=4)
            pipeline.rotate(probability=0.1, max_left_rotation=10, max_right_rotation=10)
            pipeline.shear(probability=0.1, max_shear_left=15, max_shear_right=15)
            pipeline.sample(int(samples))
            samples_counter += samples
        elif samples < 0:
            artificial_images = os.listdir(os.path.abspath(path))
            print(artificial_images)
            while samples < 0:
                os.remove(os.path.join(os.path.abspath(path), artificial_images[0]))
                if len(artificial_images) == 0:
                    break
                else:
                    artificial_images = artificial_images[1:]
                    samples += 1
                    samples_counter -= 1

    if samples_counter > 0:
        print('\nData successfully augmented with ', samples_counter, ' new samples.')
    elif samples_counter < 0:
        print('\nArtifical data augmentation reduced to given scalar. A total of ', abs(samples_counter), ' samples have been deleted.')
    else:
        print('NOTICE: Data has already been augmented. Choose higher scalar for further augmentation')
    move_files(path_to_moved_data, path_to_data, moved_paths, root=os.path.abspath(path_to_moved_data), delete_src=True)


def get_index(path_to_data):
    """
    Return path to index file of data directory.
    If no such file exists, one will be created at 'path_to_data/index.txt'
    This is done to enhance augmentation, e.g., ensure that only original
    images are used for augmentation, when applying augmentation multiple times.
    :param path_to_data: path to data to be indexed.
    :return: the path the index file
    """
    index_file_path = path_to_data + os.sep + 'index.txt'
    if not os.path.exists(index_file_path):  # TODO: actually search for index file in all folders
        print('No index file found for data set. Creating new index file..')
        lines = []
        subdirs = list(os.walk(path_to_data))[0][1]
        for i in range(len(subdirs)):
            path = os.path.join(path_to_data, subdirs[i])
            list_dir = os.listdir(path)
            for image in list_dir:
                if '.csv' in image:
                    os.remove(os.path.join(path, image))
                else:
                    path_to_img = os.path.join(path, image)
                    lines.append(str(os.path.abspath(path_to_img))+os.linesep)

        index_file_path = path_to_data + os.sep + 'index.txt'
        file = open(index_file_path, 'w')
        file.writelines(lines)
        file.close()
        # print('Index file created at ', index_file_path)
    return index_file_path


def move_files(src, dst, filenames, root='working_dir', delete_src=False):
    """
    This method moves specified files from one directory to another.
    WARNING: only works if src and dst are both (sub)+directories of the working directory!
    :param src: Source directory
    :param dst: New directory. Will be created if it doesn't exist yet.
    :param filenames: List of files given by their ABSOLUTE path.
    :param root: The root from whereon the directory will be copied to the dst folder
    :param delete_src: If this argument is True, the src directory will be deleted after the move is completed.
    :return: A list of paths to the new locations of the files
    """
    new_filenames = []
    for path in filenames:
        path = path.strip(os.linesep)
        sub_path = path.split(os.getcwd() if root == 'working_dir' else root)[1]
        new_file_path = dst + sub_path
        new_filenames.append(os.path.abspath(new_file_path))
        new_file_directory = new_file_path[:-(len(new_file_path.split(os.sep)[len(new_file_path.split(os.sep))-1]) + 1)]
        if not os.path.exists(new_file_directory):
            os.makedirs(new_file_directory)
        try:
            shutil.move(path, new_file_path)
        except:
            print('ERROR: index file doesn\'t match data! Delete index file or re-download dataset.')
            print('Aborting.')
            sys.exit(0)

    if delete_src:
        shutil.rmtree(src)

    return new_filenames


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


def convert_ppm_to_png(path_to_data):
    subdirs = list(os.walk(path_to_data))[0][1]
    for i in range(len(subdirs)):
        path = os.path.join(path_to_data, subdirs[i])
        list_dir = os.listdir(path)
        for image in list_dir:
            if '.csv' in image:
                os.remove(os.path.join(path, image))
            else:
                path_to_img = os.path.join(path, image)
                print('pnmtopng '+path_to_img+' > .'+str(path_to_img).split('.')[1]+'.png')
                os.system('pnmtopng '+path_to_img+' > .'+str(path_to_img).split('.')[1]+'.png')
                os.remove(path_to_img)


def change_image_mode_to_rgb(path_to_data):
    subdirs = list(os.walk(path_to_data))[0][1]
    for i in range(len(subdirs)):
        path = os.path.join(path_to_data, subdirs[i])
        list_dir = os.listdir(path)
        for image in list_dir:
            if '.csv' in image:
                os.remove(os.path.join(path, image))
            else:
                path_to_img = os.path.join(path, image)
                image = Image.open(path_to_img)
                # print(image.mode)
                mode = image.mode
                if mode != 'RGB':
                    print(path_to_img + ' mode: ' + str(mode))
                    image.convert('RGB').save(path_to_img)
