from data import DataLoader
from model import *
from argparse import ArgumentParser
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow as tf

import sys
tf.config.experimental_run_functions_eagerly(True)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--batch-size', default= 64, type= int)
    parser.add_argument('--train-folder', type= str)
    parser.add_argument('--valid-folder', type= str)
    parser.add_argument('--epochs', default= 10, type= int)
    parser.add_argument('--classes', default= 2, type= int)
    parser.add_argument('--lr', default= 0.001, type= float)
    parser.add_argument('--shuffle', default= True, type= bool)
    parser.add_argument('--augmented', default= True, type= bool)
    parser.add_argument('--seed', default= 2020, type= int)
    parser.add_argument('--image-size', default= 224, type= int)
    parser.add_argument('--rho', default= 1.0, type= float)
    parser.add_argument('--alpha', default= 1.0, type= float)
    parser.add_argument('--droppout', default= 0.001, type= float)
    parser.add_argument('--Mobilenetv1-folder', default= 'MobilenetV1', type= str)
    parser.add_argument('--label-smoothing', default= 0.1, type = float)
    parser.add_argument('--optimizer', default= '')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    print('---------------------Welcome to Mobilenet V1-------------------')
    print('Author')
    print('Github: Nguyendat-bit')
    print('Email: nduc0231@gmail')
    print('---------------------------------------------------------------------')
    print('Training MobileNetV1 model with hyper-params:')
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    

    # Load Data
    print("-------------LOADING DATA------------")
    datasets = DataLoader(args.train_folder, args.valid_folder, augment= args.augmented, seed= args.seed, batch_size= args.batch_size, shuffle= args.shuffle, image_size= (args.image_size, args.image_size))
    train_data, val_data = datasets.build_dataset()

    # Initializing models
    MobilenetV1 = Mobilenet_V1(classes= args.classes, alpha= args.alpha, rho= args.alpha, droppout= args.droppout, img_size= (args.image_size, args.image_size)).build()

    # Set up loss function
    loss = CategoricalCrossentropy(label_smoothing= args.label_smoothing)

    # Optimizer Definition
    if args.optimizer == 'adam':
        optimizer = Adam(learning_rate=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = SGD(learning_rate=args.lr)
    elif args.optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=args.lr)
    elif args.optimizer == 'adadelta':
        optimizer = Adadelta(learning_rate=args.lr)
    elif args.optimizer == 'adamax':
        optimizer = Adamax(learning_rate=args.lr)
    elif args.optimizer == 'adagrad':
        optimizer = Adagrad(learning_rate= args.lr)
    else:
        raise 'Invalid optimizer. Valid option: adam, sgd, rmsprop, adadelta, adamax, adagrad'

    # Callback
    checkpoint = callbacks.ModelCheckpoint(args.Mobilenetv1_folder, monitor= 'val_acc', save_best_only=  True, verbose = 1)
    lr_R = callbacks.ReduceLROnPlateau(monitor= 'acc', patience= 4, verbose= 1 , factor= 0.5, min_lr= 0.00001)

    # Complie optimizer and loss function into model
    MobilenetV1.compile(optimizer= optimizer, loss= loss, metrics= ['acc'])

    # Training model 
    print('-------------Training Mobilenet_V1------------')
    MobilenetV1.fit(train_data, validation_data= val_data, epochs= args.epochs, verbose= 1, callbacks= [checkpoint, lr_R])


