import datetime
import cv2
import os
import shutil
import sys
import csv
import zipfile
import Augmentor
import requests
from shutil import copyfile
import numpy as np
from datetime import datetime

from settings import Settings


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


def get_necessary_dataset(dataset_name, data_dir):
    """
    If data set is missing, download und unzip data set from cloud.
    :param dataset_name: name of the data set.
                         Choices: 'isf' (Custom Image Set ISF Loewen) or 'gtsrb' (German Traffic Sign Recognition Benchmark)
    :param data_dir: the directory, to which the data set will be extracted
    :return: path to the training data directory and path to the test data directory
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(data_dir + os.sep + dataset_name):
        print('Downloading ' + str(dataset_name) + ' dataset..')
        if dataset_name == 'isf':
            download_file_from_google_drive('1Xvw7w3XKNLPWwfCZMKordtS7c-sIc_cs', data_dir + os.sep + 'data.zip')
        elif dataset_name == 'isf-new':
            download_file_from_google_drive('1cD7n4HDnxbISMuFJGc6d8j7Cnqk8_6vT', data_dir + os.sep + 'data.zip')
        elif dataset_name == 'isf-complete':
            download_file_from_google_drive('1GpkTe4ryMRK2kBASHsTYiIJDaCHdWTvr', data_dir + os.sep + 'data.zip')
        elif dataset_name == 'gtsrb':
            download_file_from_google_drive('1SnQphh6TpDShavXDT6fpeyT2I9xzEa6T', data_dir + os.sep + 'data.zip')
        elif dataset_name == 'mnist':
            download_file_from_google_drive('1I5J1OZPti20w8zFGYsBqbyaiN7X8z4iB', data_dir + os.sep + 'data.zip')
        elif dataset_name == 'cifar':
            download_file_from_google_drive('1FNCLe8LRBIysWw_li7lQXVPOM_rPAb3A', data_dir + os.sep + 'data.zip')
        elif dataset_name == 'cifar100':
            download_file_from_google_drive('12iRUNoxouTPulA3FtW0vwaqOIxVmftZr', data_dir + os.sep + 'data.zip')
        else:
            print('Dataset not found and can\'t be downloaded')#
        print('Unzipping..')
        zip_ref = zipfile.ZipFile(data_dir + os.sep + 'data.zip', 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()
        print('Deleting zip..')
        os.remove(data_dir + os.sep + 'data.zip')
    return data_dir + os.sep + dataset_name + os.sep + 'train', data_dir + os.sep + dataset_name + os.sep + 'test'


def import_latest_model(import_path='.'):
    """
    Download latest model from google drive.
    If device is not connected to internet, previously downloaded models will be used, if available.
    :return: Settings of model
    """
    path_to_import = import_path+os.sep+Settings.import_dir_name
    if not internet_on():
        print('No internet connection.')
        if not os.path.exists(path_to_import):
            print('No sign-classification model has been downloaded. Please connect to internet and try again!')
            sys.exit(1)
        else:
            print('Using last downloaded sign-classification model..')
            file = open(path_to_import+os.sep+'time_of_export.txt', 'r')
            time = file.readlines()[0]
            print('Model last updated on ', time)
            settings = Settings(None, restore_from_path=path_to_import+os.sep+'settings.txt')
            settings.update({'model_save_path': os.path.abspath(path_to_import+os.sep+'model.h5')})
            return settings
    elif os.path.exists(path_to_import):
        file = open(path_to_import + os.sep + 'time_of_export.txt', 'r')
        previous_time = file.readlines()[0]
        print('Sign-classification model found, checking for updates..')
    else:
        print('No model has been downloaded.')
        print('Downloading latest sign-classification model..')
        previous_time = None

    download_file_from_google_drive('128LgaMShSbeTVG9FWxuFM1heOgGiRJOt', 'data.zip')
    zip_ref = zipfile.ZipFile('data.zip', 'r')
    zip_ref.extractall(path_to_import)
    zip_ref.close()
    os.remove('data.zip')
    file = open(path_to_import + os.sep + 'time_of_export.txt', 'r')
    time = file.readlines()[0]

    if previous_time is None:
        print('Latest model downloaded successfully. Model was uploaded to drive on ', time)
    elif datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f") > datetime.strptime(previous_time, "%Y-%m-%d %H:%M:%S.%f"):
        print('Update found. Latest model from was uploaded to drive on ', time)
    else:
        print('Model already up-to-date.')

    settings = Settings(None, restore_from_path=os.path.join(path_to_import, 'settings.txt'))
    settings.update({'model_save_path': os.path.abspath(path_to_import+os.sep+'model.h5')})

    return settings


def export_model_to_production(settings, export_path=None, overwrite=True, upload_to_drive=False):
    if export_path is None:
        export_path = settings.get_output_path() + settings.export_dir_name

    if os.path.exists(export_path):
        if overwrite:
            shutil.rmtree(export_path)
        else:
            print('ERROR: model can\'t be exorted to ', export_path, ' as this directory already exists.')
            print('Either move/delete this folder or set overwrite flag to true in export_model_to_production function call')
            print('Aborting')
            sys.exit(0)

    path_to_model = settings.get_setting_by_name('model_save_path')
    path_to_settings = settings.settings_path

    new_path_to_model = os.path.join(export_path, 'model.h5')
    new_path_to_settings = os.path.join(export_path, 'settings.txt')

    os.mkdir(export_path)
    shutil.copy(path_to_model, new_path_to_model)
    shutil.copy(path_to_settings, new_path_to_settings)
    time_of_export_file = export_path + os.sep + 'time_of_export.txt'
    file = open(time_of_export_file, 'w')
    file.writelines([datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")])
    file.close()

    shutil.make_archive(export_path, 'zip', export_path)
    shutil.rmtree(export_path)
    print('Model exported to ', export_path + '.zip')
    return new_path_to_settings


def internet_on():
    try:
        download_file_from_google_drive('1ZqDmnSt8oRbHgd7CoXsu9CYfoKusKjYK', 'hi.txt')
        os.remove('hi.txt')
        return True
    except:
        return False


def find_settings(path):
    listdir = os.listdir(path)
    for item in listdir:
        if os.path.isdir(item):
            find_settings(item)


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

    # count syntheticly added images
    subdirs = list(os.walk(path_to_data))[0][1]
    synthetic_distribution = []
    dirs = []
    for i in range(len(moved_subdirs)):
        path = os.path.join(path_to_data, subdirs[i])
        list_dir = os.listdir(path)
        synthetic_distribution.append(len(list_dir))
        dirs.append(path)

    max_samples = max(original_distribution)
    min_samples = min(original_distribution)
    samples_counter = 0

    for i in range(len(dirs)):
        moved_path = moved_dirs[i]
        path = dirs[i]

        if balance:
            if min_samples * scalar < max_samples:
                samples = max_samples - synthetic_distribution[i] - original_distribution[i]
            else:
                samples = min_samples * scalar - synthetic_distribution[i] - original_distribution[i]
        else:
            samples = original_distribution[i] * scalar - original_distribution[i] - synthetic_distribution[i]

        if samples > 0:
            # setup pipeline
            pipeline = Augmentor.Pipeline(os.path.abspath(moved_path), output_directory=(os.path.abspath(path) if output_dir == 'auto' else output_dir))
            pipeline.random_brightness(probability=0.5, min_factor=0.5, max_factor=1.5)
            pipeline.random_distortion(probability=0.1, grid_width=4, grid_height=4, magnitude=3)
            pipeline.rotate(probability=0.1, max_left_rotation=10, max_right_rotation=10)
            pipeline.shear(probability=0.2, max_shear_left=20, max_shear_right=20)
            pipeline.zoom(probability=0.2, min_factor=1.1, max_factor=1.5)
            pipeline.sample(int(samples))
            samples_counter += samples
        elif samples < 0:
            synthetic_images = os.listdir(os.path.abspath(path))
            while samples < 0:
                os.remove(os.path.join(os.path.abspath(path), synthetic_images[0]))
                if len(synthetic_images) == 0:
                    break
                else:
                    synthetic_images = synthetic_images[1:]
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


def get_file_type(path_to_file):
    """
    :param path_to_file: path to file
    :return: 'video' if file is a video
             'image' if file is an image
    """
    extension = path_to_file.split('.')[-1]
    if extension in ['avi', 'mp4', 'mp4c']:
        return 'video'
    elif extension in ['ppm', 'jpg', 'jpeg', 'png']:
        return 'image'
    elif os.path.isdir(path_to_file):
        return 'subdirectory'
    else:
        print('Warning: unsupported file type: .', extension)
        # print('aborting')
        # sys.exit(0)


def read_any_data(path_to_folder, imread_unchanged=False, settings=None, return_filenames=False):
    """
    Reads images (natively and from video) from given directory.
    :param path_to_folder: the directory containing the data.
    :param imread_unchanged: if true, read image with all given channels (including alpha)
    :param settings: transform data according to settings
    :param return_filenames: returns names of all files read in an additional array
    :return: the images as a numpy array
    """
    images = []
    names = []
    counter = 0

    listdir = os.listdir(path_to_folder)
    listdir = sorted(listdir)
    for file in listdir:
        path_to_file = os.path.join(path_to_folder, file)
        if get_file_type(path_to_file) == 'image':
            sys.stdout.write('\rreading file (image) ' + str(counter+1) + '/' + str(len(os.listdir(path_to_folder))))
            sys.stdout.flush()
            if imread_unchanged:
                image = cv2.imread(path_to_file, cv2.IMREAD_UNCHANGED)
            else:
                image = cv2.imread(path_to_file)
            if len(np.shape(image)) < 3:
                image = np.expand_dims(file, 2)
            if settings is not None:
                image = transform(image, settings)
            names.append((path_to_file.split(os.sep)[-1]).split('.')[0])
            images.append(image)
        elif get_file_type(path_to_file) == 'subdirectory':
            sys.stdout.write('reading files from subdirectory ' + path_to_file + ' ..')
            sys.stdout.flush()
            subdirectory_images = read_any_data(path_to_file, imread_unchanged, settings, return_filenames)
            if return_filenames:
                subdirectory_images, subdirectory_names = read_any_data(path_to_file, imread_unchanged, settings, return_filenames)
                for name in subdirectory_names:
                    names.append(name)
            for image in subdirectory_images:
                images.append(image)
        else:
            print('skipping file')
        counter += 1

    print("\ntotal number of images: " + str(len(images)))
    images = np.array(images)
    if return_filenames:
        return images, names
    else:
        return images


def transform(image, settings):
    """
    Transforms a given image to fit width, height and number of channels given by settings.
    :param image: image to be transformed
    :param settings: settings needed for the transformation
    :return:
    """
    if len(np.shape(image)) < 3:
        image = np.expand_dims(image, 2)

    if np.shape(image)[2] == 1:
        if settings.get_setting_by_name('channels') > 1:
            print('ERROR: data is in gray scale, yet the model requires three input channels.')
            print('No conversion from grayscale to color possible')
            print('Aborting..')
    elif np.shape(image)[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, ) if settings.get_setting_by_name('channels') == 1 else image
    elif np.shape(image)[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY, ) if settings.get_setting_by_name('channels') == 1 else image

    image = cv2.resize(image, (settings.get_setting_by_name('height'), settings.get_setting_by_name('width')))
    return image
