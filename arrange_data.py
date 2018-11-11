import os
import sys
import csv
from shutil import copyfile

src_dir = "C:\\Users\\m1a9l\\PycharmProjects\\detector\\data\\GTSRB\\test\\images"
out_dir = "C:\\Users\\m1a9l\\PycharmProjects\\detector\\data\\GTSRB\\test\\classes"
annotations = "C:\\Users\\m1a9l\\Downloads\\GTSRB_Final_Test_GT\\GT-final_test.csv"

if not os.path.exists(src_dir):
    print('src dir does not exist!')
    sys.exit(0)

if not os.path.exists(out_dir):
    os.mkdir(out_dir)


with open(annotations) as csv_file:
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
    print('Processed ', line_count-1, ' lines.')

