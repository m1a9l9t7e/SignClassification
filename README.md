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
    

--data isf --channels 1 --model transfer --restore transfer --lock cnn-dnn --restore_argument ./models/gtsrb/saves/epoch53.ckpt --epoch 200