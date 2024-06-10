from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input, Activation
from keras.optimizers import Adam
from functools import partial
import keras.backend as K

import tensorflow as tf
import tensorflow_addons as tfa
from utils import threshold_accuracy, absolute_relative_error


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
    x = (Conv2DTranspose( nClasses + nClasses1 + 7  , kernel_size=(3,3) ,  strides=(2,2) , padding='same',dilation_rate = (1,1), use_bias=False, data_format=IMAGE_ORDERING, name="Transpose_pool1" )(x))
    
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

def depth_loss (y_true, y_pred): 
    d = tf.subtract(y_pred,y_true)
    n_pixels = 128 * 128
    square_n_pixels = n_pixels * n_pixels
    square_d = tf.square(d)
    sum_square_d = tf.reduce_sum(square_d,1)
    sum_d = tf.reduce_sum(d,1)
    square_sum_d = tf.square(sum_d)
    mid_output = tf.reduce_mean((sum_square_d/n_pixels) - 0.5* (square_sum_d/square_n_pixels))

    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    
    paddings_y = tf.constant([[0,0],[1,0],[0,0],[0,0]])
    paddings_x = tf.constant([[0,0],[0,0],[1,0],[0,0]])
    
    pad_dy_true = tf.pad(dy_true, paddings_y, "CONSTANT")
    pad_dy_pred = tf.pad(dy_pred, paddings_y, "CONSTANT")
    pad_dx_true = tf.pad(dx_true, paddings_x, "CONSTANT")
    pad_dx_pred = tf.pad(dx_pred, paddings_x, "CONSTANT")

    pad_dy_true = pad_dy_true[:,:-1,:,:]
    pad_dy_pred = pad_dy_pred[:,:-1,:,:]
    pad_dx_true = pad_dx_true[:,:,:-1,:]
    pad_dx_pred = pad_dx_pred[:,:,:-1,:]

    term3 = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true) + K.abs(pad_dy_pred - pad_dy_true) + K.abs(pad_dx_pred - pad_dx_true), axis=-1)
    
    depth_output = mid_output + term3
    depth_output = K.mean(depth_output)
    return depth_output

def normal_loss(y_true, y_pred):
    assert y_pred.shape[-1] == 3 and y_true.shape[-1] == 3, "Input tensors must have three channels."


    d = tf.subtract(y_pred,y_true)
    n_pixels = 128 * 128
    square_n_pixels = n_pixels * n_pixels
    square_d = tf.square(d)
    sum_square_d = tf.reduce_sum(square_d,1)
    sum_d = tf.reduce_sum(d,1)
    square_sum_d = tf.square(sum_d)
    normal_output = tf.reduce_mean((sum_square_d/n_pixels) - 0.5* (square_sum_d/square_n_pixels))
    return normal_output 


def depth_to_normal(y_pred_depth, y_true_normal, scale_pred, scale_true):

    Scale = 127.5
    epsilon = 1e-6

    # Normalize y_true_normal
    # y_true_normal = scale_true[:, :, :, 0:1] + (scale_true[:, :, :, 1:2] - scale_true[:, :, :, 0:1]) * y_true_normal

    # Applying Laplacian filter to enhance edges
    laplacian_filter = tf.constant([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=tf.float32, shape=[3, 3, 1, 1])
    laplacian_depth = tf.nn.conv2d(y_pred_depth, laplacian_filter, strides=[1, 1, 1, 1], padding='SAME')

    # Calculate gradients from laplacian enhanced depth
    zy, zx = tf.image.image_gradients(laplacian_depth)
    zx *= Scale
    zy *= Scale

    # Create normal vectors
    normal_ori = tf.concat([-zx, -zy, tf.ones_like(y_pred_depth)], axis=-1)
    normal = normal_ori / (tf.linalg.norm(normal_ori, axis=-1, keepdims=True) + epsilon)
    normal = (normal + 1) / 2  # Normalize the output

    return normal



def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)



def scale_loss(y_true,y_pred):
    pred_depth_min = y_pred[:,0:1,0:1]
    pred_depth_max = y_pred[:,0:1,1:2]
    pred_normal_min = y_pred[:,0:1,2:3]  
    pred_normal_max = y_pred[:,0:1,3:4]

    true_depth_min = y_true[:,0:1,0:1]
    true_depth_max = y_true[:,0:1,1:2]
    true_normal_min = y_true[:,0:1,2:3]  
    true_normal_max = y_true[:,0:1,3:4]

    loss_depth_min = mean_squared_error(true_depth_min, pred_depth_min)
    loss_depth_max = mean_squared_error(true_depth_max, pred_depth_max)
    loss_normal_min = mean_squared_error(true_normal_min, pred_normal_min)
    loss_normal_max = mean_squared_error(true_normal_max, pred_normal_max)

    return tf.reduce_mean(loss_depth_min + loss_depth_max + loss_normal_min + loss_normal_max)



def combined_loss(y_true,y_pred):
    print("y_true shape:", y_true.shape)
    print("y_pred shape:", y_pred.shape)
    #print(K.int_shape(y_true)[0],K.shape(y_pred))

    depth_true = y_true[:,:,:,0]
    normal_true = y_true[:,:,:,1:4]
    img_true = y_true[:,:,:,4:7]
    ref_true = y_true[:,:,:,7:10]
    scale_true = y_true[:,:,:,10:]

    depth_pred = y_pred[:,:,:,0]
    normal_pred = y_pred[:,:,:,1:4]
    scale_pred = y_pred[:,:,:,10:]

    depth_true = tf.expand_dims(depth_true, -1)
    depth_pred = tf.expand_dims(depth_pred, -1)

    alpha = 0.2
    beta = 0.2
    gamma = 0.2
    theta = 0.2

    #depth loss
    loss_depth = alpha*(depth_loss(depth_true,depth_pred))

    #normal loss
    loss_normal = beta*(normal_loss(normal_true,normal_pred))
    
    #normal from depth
    normal_from_depth= depth_to_normal(depth_pred,normal_true,scale_pred,scale_true)
    loss_depth_to_normal = gamma*(normal_loss(normal_true,normal_from_depth)) 

    #scale_loss
    loss_scale = theta * scale_loss(scale_true,scale_pred)

    # return (loss_depth + loss_normal + loss_depth_to_normal + loss_ray_trace + loss_scale)
    return (loss_depth + loss_normal + loss_depth_to_normal + loss_scale)

def create_model():
    # Creating partial functions for each threshold
    accuracy_125 = partial(threshold_accuracy, threshold=1.25)
    accuracy_125.__name__ = 'accuracy_125'

    accuracy_15625 = partial(threshold_accuracy, threshold=1.25**2)
    accuracy_15625.__name__ = 'accuracy_15625'

    accuracy_1953125 = partial(threshold_accuracy, threshold=1.25**3)
    accuracy_1953125.__name__ = 'accuracy_1953125'

    loss_funcs = {
        "single_out": combined_loss,
    }
    loss_weights = {"single_out": 1.0}


    model = FluidNet(nClasses=1, nClasses1 = 3)  # Adjust the number of classes if necessary
    model.compile(
        optimizer='adam', 
        loss=loss_funcs, 
        loss_weights=loss_weights,
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(), 
            absolute_relative_error,
            accuracy_125,
            accuracy_15625,
            accuracy_1953125
        ]
    )
    model.summary()

    custom_objects = {
    'accuracy_125': accuracy_125,
    'accuracy_15625': accuracy_15625,
    'accuracy_1953125': accuracy_1953125,
    'combined_loss': combined_loss, 
    'absolute_relative_error': absolute_relative_error
}

    return model , custom_objects


if __name__ == "__main__":

    model = create_model()

        
