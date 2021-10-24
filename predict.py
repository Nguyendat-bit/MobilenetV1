from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import json 
import tensorflow as tf 
from argparse import ArgumentParser
import sys
import numpy as np




if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--MobilenetV1-folder', default= 'MobilenetV1',type= str)
    parser.add_argument('--test-file', type= str, required= True)
    parser.add_argument('--image-size', default= 150, type= int)
    parser.add_argument('--rho',default= 1.0, type= float)

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
    print('Predict MobileNetV1 model with hyper-params:')
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))

    # Load model
    model = load_model(args.MobilenetV1_folder)

    # Load label 
    with open('label.json') as f:
        class_indices = json.load(f)
    indices_class = dict((j,i) for i,j in class_indices.items())

    img = preprocessing.image.load_img(path= args.test_file, target_size= (args.image_size, args.image_size))
    img = preprocessing.image.img_to_array(img) / 255.
    img = tf.expand_dims(img, axis= 0) # (batch, row, col, chanel)

    result = model(img)
    result = np.argmax(result.numpy())
    print('---------------------Prediction Result: -------------------')
    print('This image is {}'.format(indices_class[result]))
