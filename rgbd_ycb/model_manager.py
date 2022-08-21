import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import layers


class ModelManager():
    def __init__(self, config):
        self.config = config
        self.w_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    
    def unet_conv(self, inputs, filters, dropout=0.4,kernel_size=3, padding="same", strides=2, activation=True):
        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer=self.w_init,
            padding=padding,
            strides=strides,
        )(inputs)
        if activation:
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Dropout(dropout)(x)
        return x 

    
    def unet_up_conv(self, inputs, filters, dropout=0.4,kernel_size=3, padding="same", strides=2, activation=True):
        x = Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer=self.w_init,
            padding=padding,
            strides=strides,
            use_bias=True
        )(inputs)
        if activation:
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Dropout(dropout)(x)
        return x

    def build_model(self, img_size=64, num_channels=3):
    
        ch_r = 4
        inputs = keras.Input(shape=(img_size,img_size,num_channels,))
        x1 = self.unet_conv(inputs, filters = 16, strides = 1)
        
        x2 = self.unet_conv(x1, filters = 4*ch_r, strides = 2)
        x3 = self.unet_conv(x2, filters = 8*ch_r, strides = 1)
        
        x4 = self.unet_conv(x3, filters = 8*ch_r, strides = 2)
        x5 = self.unet_conv(x4, filters = 16*ch_r, strides = 1)
        
        x6 = self.unet_conv(x5, filters = 16*ch_r, strides = 2)
        x7 = self.unet_conv(x6, filters = 32*ch_r, strides = 1)
        
        
        x8 = self.unet_conv(x7, filters = 32*ch_r, strides = 2)
        x9 = self.unet_conv(x8, filters = 64*ch_r, strides = 1)
        
        x10 = self.unet_up_conv(x9, filters = 64*ch_r, strides = 2)
        
        
        x = layers.concatenate([x7, x10])
        x11 = self.unet_conv(x, filters = 32*ch_r, strides = 1)
        x12 = self.unet_up_conv(x11, filters = 16*ch_r, strides = 2)
        
        x = layers.concatenate([x5, x12])
        x13 = self.unet_conv(x, filters = 16*ch_r, strides = 1)
        x14 = self.unet_up_conv(x13, filters = 8*ch_r, strides = 2)
        
        x = layers.concatenate([x3, x14])
        x15 = self.unet_conv(x, filters = 8*ch_r, strides = 1)
        x16 = self.unet_up_conv(x15, filters = 4*ch_r, strides = 2)
        
        x = layers.concatenate([x1, x16, inputs])
        x17 = self.unet_conv(x, filters = 16, strides = 1)
        
        x = layers.Conv2D(7, kernel_size = 3, padding="same")(x17)
        output = Activation("sigmoid", name="head1")(x)

        # Define the model
        model = keras.Model(inputs, outputs = [output])
        
        return model
    
    