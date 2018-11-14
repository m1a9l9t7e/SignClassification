# sign-classification
Dynamic implementation of a neural network for image classification in tensorflow from scratch.

    # If path to saved model is specified in settings file, continue training automatically
    # model.train(settings, n_epochs=args.epoch, restore_type='auto')

    # Continue train from certain epoch out of the models save directory
    # model.train(settings, n_epochs=args.epoch, restore_type='by_name', restore_data='epoch192')

    # Load model from path and start train
    # model.train(settings, n_epochs=args.epoch, restore_type='path', restore_data='./models/minimalistic-cnn/saves/epoch100.ckpt')

    # Load only certain parts of model given by path and start train. Loaded parts will be locked for train (--lock)
    # model.train(settings, n_epochs=args.epoch, restore_type='transfer', restore_data='./models/deep-cnn/saves/epoch119.ckpt')