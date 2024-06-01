from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input, Activation
from keras.optimizers import Adam
from functools import partial
import keras.backend as K

import tensorflow as tf
import tensorflow_addons as tfa
from utils import threshold_accuracy, absolute_relative_error


#create network

def FluidNet(input_height=128, input_width=128, depth_channels=1, normal_channels=3):
    IMAGE_ORDERING = "channels_last"
    img_input = Input(shape=(input_height, input_width, 6), name='combined_input')  # Assume 128x128x6 input

    # Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), name='block1_pool', data_format=IMAGE_ORDERING)(x)
    f1 = x  # Feature map 1

    # Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), name='block2_pool', data_format=IMAGE_ORDERING)(x)
    f2 = x  # Feature map 2

    # Block 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), name='block3_pool', data_format=IMAGE_ORDERING)(x)
    f3 = x  # Feature map 3

    # Block 4
    x = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), name='block4_pool', data_format=IMAGE_ORDERING)(x)
    f4 = x  # Feature map 4

    # Decoder to upsample and produce depth and normal maps
    x = Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False, data_format=IMAGE_ORDERING)(f4)
    x = concatenate([x, f3], axis=-1)  # Concatenate with feature map from Block 3

    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False, data_format=IMAGE_ORDERING)(x)
    x = concatenate([x, f2], axis=-1)  # Concatenate with feature map from Block 2

    x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False, data_format=IMAGE_ORDERING)(x)
    x = concatenate([x, f1], axis=-1)  # Concatenate with feature map from Block 1

    x = Conv2DTranspose(depth_channels + normal_channels, (4, 4), strides=(2, 2), padding='same', use_bias=False, data_format=IMAGE_ORDERING)(x)
    outputs = Activation('sigmoid', name='single_out')(x)  # Assuming sigmoid activation

    model = Model(inputs=img_input, outputs=outputs)

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

def depth_to_normal(y_pred_depth):

    Scale = 127.5
    epsilon = 1e-6

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

def get_scale_data(depth_data, normal_data):
    depth_min = tf.reduce_min(depth_data, axis=[1, 2], keepdims=True)
    depth_max = tf.reduce_max(depth_data, axis=[1, 2], keepdims=True)

    normal_min = tf.reduce_min(normal_data, axis=[1, 2], keepdims=True)
    normal_max = tf.reduce_max(normal_data, axis=[1, 2], keepdims=True)

    # Expand dimensions of depth tensors to match the shape of normal tensors
    depth_min = tf.expand_dims(depth_min, -1)  # shape becomes [?, 1, 1, 1]
    depth_max = tf.expand_dims(depth_max, -1)  # shape becomes [?, 1, 1, 1]

    scale_data = tf.concat([depth_min, depth_max, normal_min, normal_max], axis=-1)
    scale_data = tf.tile(scale_data, [1, 128, 128, 1])
    return scale_data

def combined_loss(y_true,y_pred):
    print("y_true shape:", y_true.shape)
    print("y_pred shape:", y_pred.shape)

    depth_true = y_true[:,:,:,0]
    normal_true = y_true[:,:,:,1:4] 

    depth_pred = y_pred[:,:,:,0]
    normal_pred = y_pred[:,:,:,1:4]

    scale_pred = get_scale_data(depth_pred, normal_pred)
    scale_true = get_scale_data(depth_true, normal_true)
    print("scale_true shape:", scale_true.shape)

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
    normal_from_depth = depth_to_normal(depth_pred)
    loss_depth_to_normal = gamma*(normal_loss(normal_true,normal_from_depth)) 


    #scale_loss
    loss_scale = theta * scale_loss(scale_true,scale_pred)
    print("CLEAR ON THE LOSS FUNCTIONS")

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


    model = FluidNet()  # Adjust the number of classes if necessary
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

        
