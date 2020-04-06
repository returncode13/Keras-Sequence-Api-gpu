import numpy as np
import tensorflow as tf
#import keras
#from unet import *
#from tensorflow.keras import Input
#from tensorflow.keras.optimizers import Adam


#Every Sequence must implement the __getitem__ and the __len__ methods. 
#If you want to modify your dataset between epochs you may implement on_epoch_end. 




class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self,X,Y,batch_size=32,shuffle=False):
        self.X=X
        self.Y=Y
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.on_epoch_end()
        
        
    def __len__(self):
        return int(np.floor(len(self.X)/self.batch_size))
        

    def __getitem__(self,index):
        #The method __getitem__ should return a complete batch.
        
        batch_x=self.X[index*self.batch_size:(index+1)*self.batch_size]
        batch_y=self.Y[index*self.batch_size:(index+1)*self.batch_size]
        
        return np.array(batch_x),np.array(batch_y)
        
    