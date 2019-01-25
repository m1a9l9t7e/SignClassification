import os
import sys


class Settings:
    """
    This class allows for saving and loading of settings related to the model and training process.
    New settings can be added and updated dynamically through the use of the SettingsItem class.
    During runtime settings should be retrieved via get_settings_by_name(settings_name).
    """
    separator = ';'
    models_path = '.'+os.sep+'models'+os.sep
    save_path_from_model_root = os.sep+'saves'+os.sep
    logs_path_from_model_root = os.sep+'logs'+os.sep
    output_path_from_model_root = os.sep+'output'+os.sep
    data_path_from_root = '.' + os.sep + 'data'
    export_dir_name = 'export'
    import_dir_name = 'import'
    settings = []

    def __init__(self, arg_dict, restore_from_path=None):
        if restore_from_path is None:
            for key in arg_dict:
                value = arg_dict[key]
                is_list = False
                data_type = str(type(value)).split("'")[1]
                if data_type == 'list':
                    is_list = True
                    if not value:
                        data_type = 'int'
                    else:
                        data_type = str(type(value[0])).split("'")[1]
                settings_item = SettingsItem(key, data_type, value, is_list)
                self.settings.append(settings_item)
            self.settings_path = self.get_settings_path()
            self.save(check_overwrite=True)
        else:
            self.settings_path = restore_from_path
            self.load(restore_from_path)

    def save(self, update=False, check_overwrite=False):
        """
        Saves settings as settings.txt file
        :param update: Replace old settings.
        :param check_overwrite: Check saving overwrites old settings. Take action accordingly
        :return: void
        """
        dir_path = self.get_settings_path()[:len(self.get_settings_path())-len(self.get_settings_path().split(os.sep)[-1])-1]
        # dir_path = self.get_settings_path()[:len(os.sep+'settings.txt')] easier alternative

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if update and os.path.isfile(self.settings_path):
            os.remove(self.settings_path)

        if check_overwrite and os.path.isfile(self.settings_path):
            # settings_copy = self.settings
            # self.load(self.settings_path)
            # TODO check if self.settings and settings_copy equal, if not overwrite
            # return
            # input('WARNING: This will overwriting existing settings!\nIf you want to continue, press any key. Otherwise, abort the program.')
            print('Overwriting settings')

        settings_print = []
        for settings_item in self.settings:
            settings_print.append(
                settings_item.name + self.separator + str(settings_item.value) + self.separator + settings_item.type + self.separator + str(
                    settings_item.is_list) + '\n')

        file = open(self.settings_path, 'w')
        file.writelines(settings_print)
        file.close()
        print('Settings saved at ' + self.settings_path)

    def load(self, path):
        """
        Loads settings from path.
        :param path: Path to settings.txt file
        :return: void
        """
        self.settings = []
        file = open(path, 'r')
        lines = file.readlines()
        for line in lines:
            line = line[:-1]
            line = line.split(self.separator)
            is_list = True if line[3] == 'True' else False
            if len(line[1]) == 0:
                settings_item = SettingsItem(line[0], line[2], None, is_list=is_list, read=False)
            else:
                settings_item = SettingsItem(line[0], line[2], line[1], is_list=is_list, read=True)
            self.settings.append(settings_item)

        print('Settings restored.')

    def update(self, arg_dict, write_to_file=True):
        """
        Iterates over the given dict. If settings with same name is already
        present it will be replaced, otherwise a new setting is added.
        Modified settings will be written to disk.
        :param write_to_file: write changes to file if True
        :param arg_dict: The dict with the new/updated settings.
        """
        for key in arg_dict:
            value = arg_dict[key]
            is_list = False
            data_type = str(type(value)).split("'")[1]
            if data_type == 'list':
                is_list = True
                data_type = str(type(value[0])).split("'")[1]
            new_settings_item = SettingsItem(key, data_type, value, is_list)
            exists = False
            for i in range(len(self.settings)):
                if self.settings[i].name == key:
                    self.settings[i] = new_settings_item
                    exists = True
            if not exists:
                self.settings.append(new_settings_item)
        if write_to_file:
            self.save(update=True)

    def get_setting_by_name(self, name):
        """
        Method for retrieving the value of a specific setting.
        :param name: The settings name.
        :return: The desired settings value.
        """
        for settings_item in self.settings:
            if settings_item.name == name:
                return settings_item.value
        return None

    def get_save_path(self):
        return self.models_path + self.get_setting_by_name('model_name') + self.save_path_from_model_root

    def get_logs_path(self):
        return self.models_path + self.get_setting_by_name('model_name') + self.logs_path_from_model_root

    def get_output_path(self):
        return self.models_path + self.get_setting_by_name('model_name') + self.output_path_from_model_root

    def get_settings_path(self):
        if self.settings_path is None:
            return self.models_path + self.get_setting_by_name('model_name') + os.sep + 'settings.txt'
        else:
            return self.settings_path

    def assess(self, args):
        """
        Asses settings.
        If inconsistency is found, a warning is given.
        In severe cases, the execution is aborted.
        :param args: given arguments
        :return: void
        """
        if args.restore_type == 'transfer' and self.get_setting_by_name('training_lock') == 'none':
            print('WARNING: Transfer learning attempted, but no part of old model is locked!')
            input('press any key to continue...')
        if args.dataset_name == 'mnist' and args.channels != 1:
            print('WARNING: mnist data has only one channel, but --channels was parsed as ', args.channels)
        if self.get_setting_by_name('width') * self.get_setting_by_name('height') * self.get_setting_by_name('batch_size') * 32 > 1073741824 * 4:
            print('WARNING: A single Batch exceeds 4GB of memory.')
            input('press any key to continue...')
        if len(self.get_setting_by_name('conv_filters')) != len(self.get_setting_by_name('conv_kernels')) \
                or len(self.get_setting_by_name('conv_kernels')) != len(self.get_setting_by_name('pooling_after_conv')):
            print('WARNING: cnn settings are inconsistent! conv_kernels, conv_filters and pooling_after_conv must be same length!')
            print('Aborting.')
            sys.exit(0)
        for i in range(len(self.get_setting_by_name('pooling_after_conv'))):
            width = self.get_setting_by_name('width')
            height = self.get_setting_by_name('height')
            if self.get_setting_by_name('pooling_after_conv')[i]:
                if width % 2 == 0 and height % 2 == 0:
                    width /= 2
                    height /= 2
                else:
                    print("WARNING: Pooling operations divide width or height unevenly.")
                    print('Abort.')
                    sys.exit(0)
        if self.get_setting_by_name('learning_rate_decay') < 0.9:
            print('WARNING: learning rate decay is very low (Decay is exponential!)')
            input('press any key to continue...')

        if args.restore_type == 'auto' or args.restore_type == 'AUTO':
            if self.get_setting_by_name('model_save_path') is not None:
                input_checkpoint = self.get_setting_by_name('model_save_path')
                print('Model restored from ', self.get_setting_by_name('model_save_path'))
            else:
                input_checkpoint = None
        elif args.restore_type == 'by_name':
            if not os.path.exists(self.get_save_path() + str(args.restore_argument) + '.ckpt'):
                print('ERROR: no model with name ', args.restore_argument, ' found.')
                print('Aborting.')
                sys.exit(0)
            input_checkpoint = self.get_save_path() + str(args.restore_argument) + '.ckpt'
            print('Model restored from ' + self.get_save_path() + str(args.restore_argument) + '.ckpt')
        elif args.restore_type == 'path' or args.restore_type == 'transfer':
            if not os.path.isfile(args.restore_argument + '.index'):
                print('ERROR: no model found at path: ', args.restore_argument)
                print('Aborting.')
                sys.exit(0)
            input_checkpoint = args.restore_argument
            print('Model restored from ', args.restore_argument)
        else:
            print('ERROR: invalid restore type chosen.')
            print('Aborting.')
            sys.exit(0)
        self.update({'input_checkpoint': input_checkpoint})


class SettingsItem:
    """
    Class representing a single setting via its name, value as well as its data type and whether it is a list.
    The meta data is needed to automatically save and restore the data in its correct form.
    Supported data types are: int, str, bool, float.
    """
    def __init__(self, name, type, value, is_list=False, read=False):
        self.name = name
        if type not in {'int', 'str', 'bool', 'float'}:
            print('Type ', type, ' of settings item ', name, ' not understood. Assuming string...')
            type = 'str'
        self.type = type
        self.is_list = is_list
        if not read:
            self.value = value
        else:
            self.read_value_from_string(value)

    def read_value_from_string(self, input_string):
        if self.is_list:
            value = []
            input_string = input_string[1:]
            input_string = input_string[:-1]
            if self.type == 'int':
                for element in input_string.split(','):
                    value.append(int(element))
            elif self.type == 'str':
                for element in input_string.split(','):
                    value.append(str(element))
            elif self.type == 'bool':
                for element in input_string.split(','):
                    element = element.strip(' ')
                    value.append(True if element == 'True' else False)
            elif self.type == 'float':
                for element in input_string.split(','):
                    value.append(float(element))
        else:
            if self.type == 'int':
                value = int(input_string)
            elif self.type == 'str':
                value = str(input_string)
            elif self.type == 'bool':
                input_string = input_string.strip(' ')
                value = True if input_string == 'True' else False
            elif self.type == 'float':
                value = float(input_string)
        self.value = value

    def __str__(self):
        return self.name+': '+str(self.value)+' (type: '+str(self.type)+', list: '+str(self.is_list)+')'