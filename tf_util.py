import time
import cv2
import os
import sys
import tensorflow as tf
import numpy as np

import util
from data import DataManager, BatchProvider


def freeze_graph(settings, output_dir=None):
    """
    Given a tensorflow graph, remove all nodes irrelevant to output nodes given in settings and
    convert all variables to constants.
    :param settings: settings specifying:
        - 'input_checkpoint': location of checkpoint from which model will be loaded
        - 'output_node_names': names of output nodes of graph
    :param output_dir: output directory that frozen graph will be written to
    """
    print('Freezing graph..')
    if settings.get_setting_by_name('input_checkpoint') is not None:
        input_checkpoint = settings.get_setting_by_name('input_checkpoint')
    else:
        print('ERROR: model has to first be trained/restored from checkpoint before it can be frozen!')
        print('Aborting.')
        sys.exit(0)

    output_node_names = settings.get_setting_by_name('output_node_name')
    output_dir = settings.get_output_path() if output_dir is None else output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    name = input_checkpoint.split(os.sep)[-1].split('.')[0] + 'frozen.pb'
    output_filename = os.path.join(output_dir, name)

    with tf.Session(graph=tf.Graph()) as sess:
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
        saver.restore(sess, input_checkpoint)

        # Use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the useful nodes
        )
        # Serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_filename, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print(len(output_graph_def.node), ' ops in the final graph')
        print('Frozen graph saved at ', output_filename)
        settings.update({'frozen_model_save_path': os.path.abspath(output_filename)})

    return output_graph_def


def load_graph(frozen_graph_filename):
    print('Loading graph..')
    # get graph definition
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # load graph via graph definition. name='' is important!
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    return graph


def execute_frozen_model(settings, batch_provider):

    graph = load_graph(settings.get_setting_by_name('frozen_model_save_path'))
    output_node_name = settings.get_setting_by_name('output_node_name')
    input_node_name = settings.get_setting_by_name('input_node_name')
    model_prediction_class_names = settings.get_setting_by_name('class_names')

    input_node_in_graph = False
    output_node_in_graph = False
    graph_nodes = graph.get_operations()
    print('input node name: ', graph_nodes[0].name)
    print('output node name: ', graph_nodes[-1].name)

    for op in graph.get_operations():
        if op.name == input_node_name:
            input_node_in_graph = True
        if op.name == output_node_name:
            output_node_in_graph = True
    if not (input_node_in_graph and output_node_in_graph):
        print('Input or output node specified in settings is not in graph!')
        print('Abort.')
        sys.exit(0)

    x = graph.get_tensor_by_name(input_node_name+':0')
    y = graph.get_tensor_by_name(output_node_name+':0')

    print('Executing frozen model..')
    with tf.Session(graph=graph) as sess:
        # Note: we don't nee to initialize/restore anything as there are no Variables in this graph, only hardcoded constants
        batch_x, batch_y = batch_provider.next_batch()

        while len(batch_x) > 0:
            timestamp = time.time()
            predictions = sess.run(y, feed_dict={x: batch_x})
            evaluation_time = time.time() - timestamp
            print('Evaluation time of ', evaluation_time, 'seconds, for ', len(batch_x), ' images.')

            correct = 0
            wrong = 0

            for j in range(len(predictions)):
                indicees = np.argpartition(predictions[j], -4)[-4:]
                classes = predictions[j][indicees]
                idx = np.argsort(classes)[::-1]
                classes = np.array(classes)[idx]
                indicees = np.array(indicees)[idx]
                _class_names = np.array(model_prediction_class_names)[indicees]

                print('Predictions: ')
                for k in range(len(indicees)):
                    print(str(k + 1) + '. ', _class_names[k], ' with a confidence of ', np.round(classes[k], 2))
                print('\n')

                if len(batch_y) > j:
                    print('correct class: ', batch_provider.class_names[np.argmax(batch_y[j])])
                    if np.argmax(batch_y[j]) == np.argmax(predictions[j]):
                        correct += 1
                    else:
                        wrong += 1
                cv2.imshow('image', batch_x[j])
                cv2.waitKey(0)

            batch_x, batch_y = batch_provider.next_batch()
    print('Execution finished.')


def execute_on_subimages(settings, images, rectslist):
    graph = load_graph(settings.get_setting_by_name('frozen_model_save_path'))
    output_node_name = settings.get_setting_by_name('output_node_name')
    input_node_name = settings.get_setting_by_name('input_node_name')
    class_names = settings.get_setting_by_name('class_names')

    input_node_in_graph = False
    output_node_in_graph = False
    graph_nodes = graph.get_operations()

    for op in graph.get_operations():
        if op.name == input_node_name:
            input_node_in_graph = True
        if op.name == output_node_name:
            output_node_in_graph = True
    if not (input_node_in_graph and output_node_in_graph):
        print('Input or output node specified in settings is not in graph!')
        print('Abort.')
        sys.exit(0)

    input_node = graph.get_tensor_by_name(input_node_name+':0')
    output_node = graph.get_tensor_by_name(output_node_name+':0')

    with tf.Session(graph=graph) as sess:

        batch_size = settings.get_setting_by_name('batch_size')
        for i in range(len(images)):
            image = images[i]
            rects = np.copy(rectslist[i])
            # for rect in rects:
            #     print(rect)
            subimages = []
            predictions = []
            while len(rects) > 0:
                if len(rects) > batch_size:
                    batch_rects = rects[0:batch_size]
                    rects = rects[batch_size+1:]
                else:
                    batch_rects = rects
                    rects = []

                batch_subimages = []

                for rect in batch_rects:
                    x, y, width, height = rect.unpack()
                    subimage = image[y:y + height, x:x + width]
                    if settings.get_setting_by_name('channels') == 1:
                        subimage = cv2.cvtColor(subimage, cv2.COLOR_BGR2GRAY, )
                    subimage = cv2.resize(subimage, (settings.get_setting_by_name('height'), settings.get_setting_by_name('width')))
                    if len(np.shape(subimage)) < 3:
                        subimage = np.expand_dims(subimage, 3)
                    batch_subimages.append(subimage)

                batch_prediction = sess.run(output_node, feed_dict={input_node: batch_subimages})
                # predictions.append(batch_prediction)
                # subimages.append(batch_subimages)
                if len(predictions) == 0:
                    predictions = batch_prediction
                    subimages = batch_subimages
                else:
                    predictions = np.concatenate([predictions, batch_prediction])
                    subimages = np.concatenate([subimages, batch_subimages])

            if not(len(predictions) == len(subimages)):
                print('ERROR in subimage loop!')
                print('len(predictions): ', len(predictions), ' len(subimages): ', len(subimages))
                sys.exit(0)

            rects = rectslist[i]
            # print('just before rects: ', np.shape(rects))
            # print('just before predictions: ', np.shape(predictions))
            for j in range(len(predictions)):
                np.set_printoptions(linewidth=300)
                indicees = np.argpartition(predictions[j], -4)[-4:]
                classes = predictions[j][indicees]
                idx = np.argsort(classes)[::-1]
                classes = np.array(classes)[idx]
                indicees = np.array(indicees)[idx]
                _class_names = np.array(class_names)[indicees]

                print('Predictions: ')
                for k in range(len(indicees)):
                    print(str(k+1)+'. ', _class_names[k][2:-1], ' with a confidence of ', np.round(classes[k], 2))
                print('\n')

                x, y, width, height = rects[j].unpack()
                image_with_rect = cv2.rectangle(np.copy(image), (x, y), (width, height), (0, 255, 0), 2)
                cv2.imshow('image', image_with_rect)
                cv2.waitKey(0)
    print('Execution finished.')


# import uff
# def convert_frozen_to_uff(settings, output_dir=None):
#     output_dir = settings.get_output_path() if output_dir is None else output_dir
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     name = settings.get_setting_by_name('frozen_model_save_path').split(os.sep)[-1].split('.')[0] + 'frozen.uff'
#     output_filename = os.path.join(output_dir, name)
#
#     converted = uff.from_tensorflow_frozen_model(frozen_file=settings.get_setting_by_name('frozen_model_save_path'),
#                                                  output_nodes=[settings.get_setting_by_name('output_node_name')], preprocessor=None)
#     with tf.gfile.GFile(output_filename, "wb") as f:
#         f.write(converted)
