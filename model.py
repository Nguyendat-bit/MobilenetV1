from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.models import Model

class Mobilenet_V1(Model):
    def __init__(self,*, classes, alpha = 1.0, rho = 1.0, droppout = 0.001):
        super(Mobilenet_V1, self).__init__()
        assert alpha > 0 and alpha <= 1 ,'Error, my Mobilenet_V1 can only accept  alpha > 0 and alpha <= 1'
        assert rho > 0 and rho <= 1 ,'Error, my Mobilenet_V1 can only accept  rho > 0 and rho <= 1'
        self._alpha = alpha
        self._rho = rho
        self._num_classes = classes
        self._droppout = droppout

    def __Standard_Conv(self):
        return models.Sequential([
            Conv2D(filters= 32, kernel_size=(3,3), strides= (1,1), padding= 'valid'),
            BatchNormalization(),
            Activation('relu')
        ])
    
    def __Depthwise_Conv(self, strides):
        return models.Sequential([
            DepthwiseConv2D(kernel_size= (3,3), strides= strides, padding= 'same' if  strides == (1,1) else 'valid'),
            BatchNormalization(),
            Activation('relu')
        ])
    
    def __Pointwise_Conv(self, filters):
        return models.Sequential([
            Conv2D(filters= int(filters * self._alpha), kernel_size= (1,1), strides= 1),
            BatchNormalization(),
            Activation('relu')
            
        ])

    def __Depthwise_Separable_Conv( self,*, strides_depthwise, filters_pointwise):
        return models.Sequential([
            self.__Depthwise_Conv(strides= strides_depthwise),
            self.__Pointwise_Conv(filters= filters_pointwise)
        ])

    def call(self,images):

        featur_map_size = int(images.shape[-1] * self._rho) 
        images = InputLayer(input_shape= (featur_map_size,featur_map_size))(images)

        result =  self.__Standard_Conv()(images)

        # Depth_Separable_Conv 1
        result = self.__Depthwise_Separable_Conv(strides_depthwise= (1,1), filters_pointwise= 64)(result)
        # Depth_Separable_Conv 2
        result = self.__Depthwise_Separable_Conv(strides_depthwise= (2,2), filters_pointwise= 128)(result)
        # Depth_Separable_Conv 3
        result = self.__Depthwise_Separable_Conv(strides_depthwise= (1,1), filters_pointwise= 128)(result)
        # Depth_Separable_Conv 4
        result = self.__Depthwise_Separable_Conv(strides_depthwise= (2,2), filters_pointwise= 256)(result)
        # Depth_Separable_Conv 5
        result = self.__Depthwise_Separable_Conv(strides_depthwise= (1,1), filters_pointwise= 256)(result)
        # Depth_Separable_Conv 6
        result = self.__Depthwise_Separable_Conv(strides_depthwise= (2,2), filters_pointwise= 512)(result)
        # Depth_Separable_Conv 7 - > 11
        for i in range(5):
            result = self.__Depthwise_Separable_Conv(strides_depthwise= (1,1), filters_pointwise= 512)(result)
        # Depth_Separable_Conv 12
        result = self.__Depthwise_Separable_Conv(strides_depthwise= (2,2), filters_pointwise= 1024)(result)

        result = ZeroPadding2D(padding= (4,4))(result)

        # Depth_Separable_Conv 13
        result = self.__Depthwise_Separable_Conv(strides_depthwise= (2,2), filters_pointwise= 1024)(result)
        result = GlobalAveragePooling2D()(result)

        # FC
        result = Dropout(self._droppout)(result)
        return Dense(self._num_classes, activation= 'softmax')(result)

    

