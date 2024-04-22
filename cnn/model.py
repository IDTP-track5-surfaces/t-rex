from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input, Activation
from keras.optimizers import Adam

from utils import combined_loss




#create network
def FluidNet( nClasses, nClasses1 ,  input_height=128, input_width=128):
    assert input_height%32 == 0
    assert input_width%32 == 0
    IMAGE_ORDERING =  "channels_last" 

    img_input = Input(shape=(input_height,input_width, 6), name='combined_input') ## Assume 128,128,6
    
    ## Block 1 128x128
    x = Conv2D(18, (2, 2), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
    x = Conv2D(18, (2, 2), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
    f1 = x
    
    # Block 2 64x64
    x = Conv2D(36, (2, 2), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(36, (2, 2), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
    f2 = x

    # Block 3 32x32
    x = Conv2D(72, (2, 2), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(72, (2, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(72, (2, 2), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
    pool3 = x

    # Block 4 16x16
    x = Conv2D(144, (2, 2), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(144, (2, 2), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(144, (2, 2), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)

    # Block 5 8x8
    x = Conv2D(144, (2, 2), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(pool4)
    x = Conv2D(144, (2, 2), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(144, (2, 2), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)
    
    # Block Transpose <DECODER> : Depth
    #1st deconv layer 4x4
    x = (Conv2DTranspose( 72, kernel_size=(4,4) ,  strides=(2,2) , padding='same', dilation_rate = (1,1), use_bias=False, data_format=IMAGE_ORDERING, name="Transpose_pool5" ) (pool5))
   
    #concatinate x and pool4 for 2nd Deconv layer 8X8
    x = concatenate ([x, pool4],axis = 3)
    x = (Conv2DTranspose( 36 , kernel_size=(6,6) ,  strides=(2,2) ,padding='same', dilation_rate = (1,1), use_bias=False, data_format=IMAGE_ORDERING, name="Transpose_pool4")(x))
    
    #concatinate x and pool3 for 3rd Deconv layer 28x28
    x = concatenate ([x, pool3],axis = 3)    
    x= (Conv2DTranspose( 18 , kernel_size=(4,4) ,  strides=(2,2) , padding='same',dilation_rate = (1,1), use_bias=False, data_format=IMAGE_ORDERING, name="Transpose_pool3" )(x))
    
    #concatinate x and f2 for 4th Deconv layer
    x = concatenate ([x, f2],axis = 3)    
    x = (Conv2DTranspose( 9 , kernel_size=(4,4) ,  strides=(2,2) ,  padding='same',dilation_rate = (1,1), use_bias=False, data_format=IMAGE_ORDERING, name="Transpose_pool2" )(x))
    
    #concatinate x and f1 for 5th Deconv layer
    
    x = concatenate ([x, f1],axis = 3)    
    x = (Conv2DTranspose( nClasses + nClasses1 , kernel_size=(3,3) ,  strides=(2,2) , padding='same',dilation_rate = (1,1), use_bias=False, data_format=IMAGE_ORDERING, name="Transpose_pool1" )(x))
    
    o = x
    o = (Activation('sigmoid', name="depth_out"))(o)

    # Block Transpose <DECODER> : Scale
    #1st deconv layer 7x7
    x2 = (Conv2DTranspose( 72, kernel_size=(4,4) ,  strides=(2,2) , padding='same', dilation_rate = (1,1), use_bias=False, data_format=IMAGE_ORDERING, name="Transpose_pool5_2" ) (pool5))
   
    #concatinate x and pool4 for 2nd Deconv layer 14x14
    x2 = concatenate ([x2, pool4],axis = 3)
    x2 = (Conv2DTranspose( 36 , kernel_size=(6,6) ,  strides=(2,2) ,padding='same', dilation_rate = (1,1), use_bias=False, data_format=IMAGE_ORDERING, name="Transpose_pool4_2")(x2))
    
    #concatinate x and pool3 for 3rd Deconv layer 28x28
    x2 = concatenate ([x2, pool3],axis = 3)    
    x2= (Conv2DTranspose( 18 , kernel_size=(4,4) ,  strides=(2,2) , padding='same',dilation_rate = (1,1), use_bias=False, data_format=IMAGE_ORDERING, name="Transpose_pool3_2" )(x2))
    
    #concatinate x and f2 for 4th Deconv layer
    x2 = concatenate ([x2, f2],axis = 3)    
    x2 = (Conv2DTranspose( 9 , kernel_size=(4,4) ,  strides=(2,2) ,  padding='same',dilation_rate = (1,1), use_bias=False, data_format=IMAGE_ORDERING, name="Transpose_pool2_2" )(x2))
    
    #concatinate x and f1 for 5th Deconv layer
    
    x2 = concatenate ([x2, f1],axis = 3)    
    x2 = (Conv2DTranspose( 7 , kernel_size=(3,3) ,  strides=(2,2) , padding='same',dilation_rate = (1,1), use_bias=False, data_format=IMAGE_ORDERING, name="Transpose_pool1_2" )(x2))
    
    o2 = x2
    o2 = (Activation('sigmoid', name="scale_out"))(o2)

    singleOut = concatenate([o,o2],axis = 3, name="single_out")

    #model creation
    model = Model(img_input, singleOut)
       
    return model

def create_model():
    model = FluidNet(nClasses=1, nClasses1=3)  # Adjust the number of classes based on your task

    loss_funcs = {
        "single_out": combined_loss,
    }

    loss_weights = {
        "single_out": 1.0,
    }
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=loss_funcs,
                  loss_weights=loss_weights,  
                  metrics=['accuracy'])
    model.summary()
    return model


if __name__ == "__main__":

    model = create_model()

        
