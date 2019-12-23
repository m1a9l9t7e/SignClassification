import cv2
import numpy as np
from models.resnet_101 import resnet101_model as build_resnet
from models.inception_v4 import inception_v4_model as build_inception


def train(settings, data_manager, restore_model=None):
    # Construct model and load imagenet weights
    architecture = settings.get_setting_by_name('model_architecture')
    backbone_trainable = not settings.get_setting_by_name('freeze_backbone')
    if architecture == 'inception':
        model = build_inception(settings, load_imagenet_weights=(restore_model is None), trainable=backbone_trainable)
    elif architecture == 'resnet':
        model = build_resnet(settings, load_imagenet_weights=(restore_model is None))
    else:
        print('Error: model architecture ' + architecture + ' is currently not supported')
        return None

    # Continue training if custom restore weights are specified
    if restore_model is not None:
        model.load_weights(restore_model, by_name=True)

    # Start Fine-tuning
    model.fit_generator(data_manager.yield_train_batch(settings.get_setting_by_name('batch_size')),
                        nb_epoch=settings.get_setting_by_name('epoch'),
                        samples_per_epoch=data_manager.get_number_train_samples(),
                        verbose=1,
                        validation_data=data_manager.yield_test_batch(settings.get_setting_by_name('batch_size')),
                        nb_val_samples=data_manager.get_number_test_samples()
                        )

    return model


def evaluate_images_model_loaded(settings, model, images, labels=None, show=True):
    predictions_valid = model.predict(images, batch_size=settings.get_setting_by_name('batch_size'), verbose=1)
    predictions = []

    for i in range(len(images)):

        predicted_class = settings.get_setting_by_name('class_names')[np.argmax(predictions_valid[i])]
        predictions.append(predicted_class)

        if not show:
            continue

        indicees = np.argpartition(predictions_valid[i], -4)[-4:]
        classes = predictions_valid[i][indicees]
        idx = np.argsort(classes)[::-1]
        classes = np.array(classes)[idx]
        indicees = np.array(indicees)[idx]
        _class_names = np.array(settings.get_setting_by_name('class_names'))[indicees]

        print('Predictions: ')
        for k in range(len(indicees)):
            print(str(k + 1) + '. ', _class_names[k], ' with a confidence of ', np.round(classes[k], 2))

        if labels is not None:
            # print(predicted_class, ' vs ', settings.get_setting_by_name('class_names')[np.argmax(labels[i])])
            prediction_correct = predicted_class == settings.get_setting_by_name('class_names')[np.argmax(labels[i])]
            image = cv2.putText(images[i], predicted_class, (0, settings.get_setting_by_name('height') // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if prediction_correct else (0, 0, 255),
                                thickness=2)
        else:
            image = cv2.putText(images[i], predicted_class, (0, settings.get_setting_by_name('height') // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), thickness=2)
        # if labels is not None and not prediction_correct:
        cv2.imshow('results', image)
        cv2.waitKey(0)

    return predictions


def evaluate_images(settings, path_to_model, images, labels=None, show=True):
    architecture = settings.get_setting_by_name('model_architecture')
    if architecture == 'inception':
        model = build_inception(settings, load_imagenet_weights=False)
    elif architecture == 'resnet':
        model = build_resnet(settings, load_imagenet_weights=False)
    else:
        print('Error: model architecture ' + architecture + ' is not supported')
        return None
    model.load_weights(path_to_model, by_name=True)
    evaluate_images_model_loaded(settings, model, images, labels=labels, show=show)
