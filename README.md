# Sign Classification

The purpose of this project is to accurately classify images of traffic sign for the annual Carolo Cup.
The program is only designed to classify single pre-cropped images and does not perform object detection. 

## Setup

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
### Prerequisites

What things you need to install beforehand

```
Python 3
(the project was tested with Python 3.5)
```

```
CUDA, in the case you want to use/have an nvidia gpu
(the project was tested with CUDA 9)
```

```
Linux is not a requirement for this project
(the project was tested on both linux and windows)
```

### Installing

The setup for this project is fairly simple.

```
Install the python packages listed in requirements.txt
```

Additionally, you will need to install the tensorflow python package, depending on your hardware.
```
If you have an Nvidia GPU and CUDA set up, install tensorflow-gpu
Otherwise, install tensorflow
```
Note that the tensorflow version is important, the correct version to use is specified in requirements.txt 

If you want to train a model, you also need to download the necessary imagenet weights

* Download weights for resnet-101 [here](https://drive.google.com/file/d/0Byy2AcGyEVxfTmRRVmpGWDczaXM/view)
* Download weights for inception-v4 [here](https://github.com/kentsommer/keras-inceptionV4/releases/download/2.0/inception-v4_weights_tf_dim_ordering_tf_kernels.h5)
* More weights for different models can be found [here](https://github.com/flyyufelix/cnn_finetune#imagenet-pretrained-models), however these are not supported as of yet
```
The downloaded weights must be saved in the imagenet_weights folder
```

## Getting started

This section will show you how to train a model or use an existing model to classify new images.

### Training a model

```
execute train.py
```

You can use a range of options for training, however the defaults are setup in a way that training should work out of the box.

If you want to save the model after training and execute it on the test data, you need to supply the following options

```
execute train.py --save --execute
```


You don't need to worry about training data, the program 
will automatically download it for you if it's not available.

You can change which data set will be downloaded by adjusting the --dataset option.
Using the following should yield the best results

```
--dataset isf-complete
```

Here are some parameters to adjust training

* epochs
```
The number of epochs to train for. This parameter is important to avoid over/underfitting.
One epoch uses all available training material
```

* batch_size

```
Choosing a higher batch size speeds up training, however you
need to stay within the limits of you gpu memory. A typical value would be 8
```

### Executing a model

```
execute test.py
```

Here you only need to specify 
* The path to the settings of trained model
* The path to a folder with images to be classified 

E.g.


```
test.py --settings ./training_results/my_model/settings.txt --data path/to/images
```