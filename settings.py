import os


class Settings:
    separator = ';'
    logs_path = '.'+os.sep+'logs'+os.sep
    models_path = '.'+os.sep+'models'+os.sep
    settings_path = None

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
                # print(settings_item)
                self.settings.append(settings_item)
            self.save(check_overwrite=True)
        else:
            self.load(restore_from_path)

    def save(self, update=False, check_overwrite=False):
        settings_path = self.models_path + self.get_setting_by_name('model_name') + os.sep + 'settings.txt'
        dir_path = self.models_path + self.get_setting_by_name('model_name')

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if update and os.path.isfile(settings_path):
            os.remove(settings_path)

        if check_overwrite and os.path.isfile(settings_path):
            # settings_copy = self.settings
            # self.load(settings_path)
            # TODO check if self.settings and settings_copy equal, if not overwrite
            # return
            print('overwriting existing settings...')

        settings_print = []
        for settings_item in self.settings:
            settings_print.append(
                settings_item.name + self.separator + str(settings_item.value) + self.separator + settings_item.type + self.separator + str(
                    settings_item.is_list) + '\n')

        file = open(settings_path, 'w')
        file.writelines(settings_print)
        file.close()
        print('settings saved at ' + settings_path)
        self.settings_path = settings_path

    def load(self, path):

        self.settings = []
        file = open(path, 'r')
        lines = file.readlines()
        for line in lines:
            line = line[:-1]
            line = line.split(self.separator)
            is_list = True if line[3] == 'True' else False
            settings_item = SettingsItem(line[0], line[2], line[1], is_list=is_list, read=True)
            self.settings.append(settings_item)

        print('settings restored.')

    def update(self, arg_dict):
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
        self.save(update=True)

    def get_setting_by_name(self, name):
        for settings_item in self.settings:
            if settings_item.name == name:
                return settings_item.value
        return None

    def print(self):
        for setting in self.settings:
            print(setting)


class SettingsItem:
    def __init__(self, name, type, value, is_list=False, read=False):
        self.name = name
        if type not in {'int', 'str', 'bool', 'float'}:
            print('type ', type, ' of settings item ', name, ' not understood. Assuming string...')
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