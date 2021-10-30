from tensorflow.keras.layers import *
from tensorflow.keras import *

class Mobilenet_V1():
    def __init__(self,*, classes, alpha = 1.0, rho = 1.0, droppout = 0.001, img_size = (224,224)):
        assert alpha > 0 and alpha <= 1 ,'Error, my Mobilenet_V1 can only accept  alpha > 0 and alpha <= 1'
        assert rho > 0 and rho <= 1 ,'Error, my Mobilenet_V1 can only accept  rho > 0 and rho <= 1'
        self._alpha = alpha
        self._rho = rho
        self._num_classes = classes
        self.model = None 
        self._droppout = droppout
        self.img_size = img_size
    def __Standard_Conv(self):
        return models.Sequential([
            Conv2D(filters= 32, kernel_size=(3,3), strides= (2,2), padding= 'valid'),
            BatchNormalization(),
            ReLU(max_value= 6)
        ])
    
    def __Depthwise_Conv(self, strides, padding):
        return models.Sequential([
            DepthwiseConv2D(kernel_size= (3,3), strides= strides, padding= padding),
            BatchNormalization(),
            ReLU(max_value= 6)
        ])
    
    def __Pointwise_Conv(self, filters):
        return models.Sequential([
            Conv2D(filters= int(filters * self._alpha), kernel_size= (1,1), strides= 1),
            BatchNormalization(),
            ReLU(max_value= 6)
            
        ])

    def __Depthwise_Separable_Conv( self,*, strides_depthwise, padding_depthwise, filters_pointwise):
        return models.Sequential([
            self.__Depthwise_Conv(strides= strides_depthwise, padding= padding_depthwise),
            self.__Pointwise_Conv(filters= filters_pointwise)
        ])

    def build(self):

        featur_map_size = int(self.img_size[0] * self._rho) 

        self.model = Sequential([
            InputLayer(input_shape= (featur_map_size,featur_map_size,3)),

            self.__Standard_Conv(),

        # Depth_Separable_Conv 1
            self.__Depthwise_Separable_Conv(strides_depthwise= (1,1), padding_depthwise= 'same',filters_pointwise= 64),
        # Depth_Separable_Conv 2
            self.__Depthwise_Separable_Conv(strides_depthwise= (2,2), padding_depthwise= 'valid', filters_pointwise= 128),
        # Depth_Separable_Conv 3
            self.__Depthwise_Separable_Conv(strides_depthwise= (1,1), padding_depthwise= 'same', filters_pointwise= 128),
        # Depth_Separable_Conv 4
            self.__Depthwise_Separable_Conv(strides_depthwise= (2,2), padding_depthwise= 'valid', filters_pointwise= 256),
        # Depth_Separable_Conv 5
            self.__Depthwise_Separable_Conv(strides_depthwise= (1,1), padding_depthwise= 'same', filters_pointwise= 256),
        # Depth_Separable_Conv 6
            self.__Depthwise_Separable_Conv(strides_depthwise= (2,2), padding_depthwise= 'valid', filters_pointwise= 512),
        # Depth_Separable_Conv 7 - > 11
                self.__Depthwise_Separable_Conv(strides_depthwise= (1,1), padding_depthwise= 'same', filters_pointwise= 512),
                self.__Depthwise_Separable_Conv(strides_depthwise= (1,1), padding_depthwise= 'same', filters_pointwise= 512),
                self.__Depthwise_Separable_Conv(strides_depthwise= (1,1), padding_depthwise= 'same', filters_pointwise= 512),
                self.__Depthwise_Separable_Conv(strides_depthwise= (1,1), padding_depthwise= 'same', filters_pointwise= 512),
                self.__Depthwise_Separable_Conv(strides_depthwise= (1,1), padding_depthwise= 'same', filters_pointwise= 512),
        # Depth_Separable_Conv 12
             self.__Depthwise_Separable_Conv(strides_depthwise= (2,2), padding_depthwise= 'valid', filters_pointwise= 1024),

        # Depth_Separable_Conv 13
            self.__Depthwise_Separable_Conv(strides_depthwise= (2,2), padding_depthwise= 'same', filters_pointwise= 1024),
            GlobalAveragePooling2D(),
    
        # FC
            Dropout(self._droppout),
            Dense(self._num_classes, activation= 'softmax')
        ])
        return self.model
        
    

