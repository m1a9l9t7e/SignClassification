To run this project the following packages have to be installed:

python-opencv
Pillow
Augmentor
ascii-graph

# sign-classification
Dynamic implementation of a neural network for image classification in tensorflow from scratch.

Options for restoring a model:
- If path to saved model is specified in settings file, continue training automatically
    - restore_type='auto'

- Continue train from certain epoch out of the models save directory
    - restore_type='by_name'
    - restore_argument='epoch192'

- Load model from path and start train
    - restore_type='path', 
    - restore_argument='./models/minimalistic-cnn/saves/epoch100.ckpt'

- Load only certain parts of model given by path and start train. These parts can also be excluded from training
    - restore_type='transfer',
    - restore_argument='./models/deep-cnn/saves/epoch119.ckpt'
    - lock='cnn'
    

Usage Examples:

Train using transfer learning where cnn and dnn are transfered/locked:

--data isf --channels 1 --model transfer --restore transfer --lock cnn-dnn --restore_argument ./models/gtsrb/saves/epoch53.ckpt --epoch 200

Train with synthetic data:

--synthetic_data --background ./data/sliding_window --foreground ./data/signs_clean --model synthetic-test --width 64 --height 64 --freeze --epoch 20 --execute --channels 1