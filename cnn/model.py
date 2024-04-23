from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input, Activation
from keras.optimizers import Adam
from functools import partial

import tensorflow as tf

def depth_loss(y_true, y_pred):
    # Squared differences
    diff = y_pred - y_true
    diff_squared = tf.square(diff)
    l2_loss = tf.reduce_mean(diff_squared)
    
    # Scale-invariant term
    scale_invariant = tf.reduce_sum(diff) ** 2
    scale_invariant_loss = scale_invariant / (2 * tf.cast(tf.size(diff), tf.float32) ** 2)
    
    # Gradient loss
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    dy_diff = dy_pred - dy_true
    dx_diff = dx_pred - dx_true
    gradient_loss = tf.reduce_mean(tf.square(dy_diff)) + tf.reduce_mean(tf.square(dx_diff))
    
    # Combine losses
    return l2_loss - scale_invariant_loss + gradient_loss


def FluidNet(nClasses = 1, nClasses1=3, input_height=128, input_width=128):
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    IMAGE_ORDERING = "channels_last"

    img_input = Input(shape=(input_height, input_width, 6), name='combined_input')

    # Encoder Blocks
    x = Conv2D(18, (2, 2), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(18, (2, 2), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING)(x)
    f1 = x

    x = Conv2D(36, (2, 2), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(36, (2, 2), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING)(x)
    f2 = x

    x = Conv2D(72, (2, 2), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(72, (2, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(72, (2, 2), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING)(x)
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
    output = (Activation('sigmoid', name="depth_out"))(o)

    # Model creation
    model = Model(inputs=img_input, outputs=output)
    return model

def threshold_accuracy(y_true, y_pred, threshold=1.25):
    ratio = tf.maximum(y_true / y_pred, y_pred / y_true)
    return tf.reduce_mean(tf.cast(ratio < threshold, tf.float32))

def absolute_relative_error(y_true, y_pred):
    # Avoid division by zero
    y_true = tf.cast(y_true, dtype=tf.float32)
    epsilon = tf.keras.backend.epsilon()  # Small constant for numerical stability
    # Calculate Absolute Relative Error
    are = tf.abs((y_true - y_pred) / (y_true + epsilon))
    return tf.reduce_mean(are)

def create_model():
    # Creating partial functions for each threshold
    accuracy_125 = partial(threshold_accuracy, threshold=1.25)
    accuracy_125.__name__ = 'accuracy_125'

    accuracy_15625 = partial(threshold_accuracy, threshold=1.25**2)
    accuracy_15625.__name__ = 'accuracy_15625'

    accuracy_1953125 = partial(threshold_accuracy, threshold=1.25**3)
    accuracy_1953125.__name__ = 'accuracy_1953125'


    model = FluidNet(nClasses=1)  # Adjust the number of classes if necessary
    model.compile(
        optimizer='adam', 
        loss=depth_loss, 
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
    'depth_loss': depth_loss,  # Assuming combined_loss is your custom loss function
    'absolute_relative_error': absolute_relative_error
}

    return model , custom_objects


if __name__ == "__main__":

    model = create_model()

        
