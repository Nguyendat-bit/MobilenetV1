import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataLoader():
    def __init__(self, Data_folder , Validation_Data_folder , augment:bool = True, seed:int = None, batch_size:int = 64, shuffle:bool = True, image_size = (224,224)):

        assert Data_folder != None, 'Error, Data_folder is not empty !'
        assert Validation_Data_folder != None, 'Error, Validation_data_folder is not empty !'

        self.seed = seed
        self.augement = augment
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_folder = Data_folder
        self.val_data_folder = Validation_Data_folder
        self.img_size = image_size

    def __gen_new_img(self,data_folder :str, augment :bool):
        if augment:
            data_gen = ImageDataGenerator(
                rescale= 1. /255,
                rotation_range= 20, 
                vertical_flip= True,
                horizontal_flip= True,
                fill_mode= 'nearest'
            )
        else:
            data_gen = ImageDataGenerator(rescale= 1. /255)
        
        data = data_gen.flow_from_directory(
            data_folder,
            target_size= self.img_size,
            batch_size= self.batch_size,
            shuffle= self.shuffle,
            class_mode= 'categorical',
            seed= self.seed
        )
        return data
    
    def build_dataset(self):
        return self.__gen_new_img(self.data_folder,augment= self.augement), self.__gen_new_img(self.val_data_folder, augment= False)
