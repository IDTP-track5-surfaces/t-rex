from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input, Activation
from functools import partial

import tensorflow as tf
from utils import threshold_accuracy, absolute_relative_error, root_mean_squared_error
from scipy.interpolate import bisplrep, bisplev
import numpy as np

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

    term3 = tf.reduce_mean(tf.abs(dy_pred - dy_true) + tf.abs(dx_pred - dx_true) + tf.abs(pad_dy_pred - pad_dy_true) + tf.abs(pad_dx_pred - pad_dx_true), axis=-1)
    
    depth_output = mid_output + term3
    depth_output = tf.reduce_mean(depth_output)
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
    return tf.reduce_mean(tf.square(y_pred - y_true), axis=-1)

def scale_loss(depth_true, depth_pred, normal_true, normal_pred):

    # Calculate min and max for depth
    depth_min_true = tf.reduce_min(depth_true, axis=[1, 2], keepdims=True)
    depth_max_true = tf.reduce_max(depth_true, axis=[1, 2], keepdims=True)
    depth_min_pred = tf.reduce_min(depth_pred, axis=[1, 2], keepdims=True)
    depth_max_pred = tf.reduce_max(depth_pred, axis=[1, 2], keepdims=True)


    # Calculate min and max for normals
    normal_min_true = tf.reduce_min(normal_true, axis=[1, 2], keepdims=True)
    normal_max_true = tf.reduce_max(normal_true, axis=[1, 2], keepdims=True)
    normal_min_pred = tf.reduce_min(normal_pred, axis=[1, 2], keepdims=True)
    normal_max_pred = tf.reduce_max(normal_pred, axis=[1, 2], keepdims=True)

    # Calculate MSE for depth and normal ranges
    loss_depth_min = tf.reduce_mean(tf.square(depth_min_pred - depth_min_true))
    loss_depth_max = tf.reduce_mean(tf.square(depth_max_pred - depth_max_true))
    loss_normal_min = tf.reduce_mean(tf.square(normal_min_pred - normal_min_true))
    loss_normal_max = tf.reduce_mean(tf.square(normal_max_pred - normal_max_true))

    # Combine the losses
    return tf.reduce_mean(loss_depth_min + loss_depth_max + loss_normal_min + loss_normal_max)

def create_knots(k, n, min_val, max_val):
    """Create knots for B-spline."""
    return np.concatenate([
        np.full(k, min_val),
        np.linspace(min_val, max_val, n - 1),
        np.full(k, max_val)
    ])

def fit_Bspline(x, y, z, k, tx, ty):
    """Fit a B-spline and return the knot vector (tck). Ensure inputs are numpy arrays."""
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and isinstance(z, np.ndarray), "Inputs must be numpy arrays"
    assert len(x) == len(y) == len(z), "Length of x, y, and z must be the same"
    tck, total_error, _, _ = bisplrep(x, y, z, kx=k, ky=k, task=-1, s=0, tx=tx, ty=ty, full_output=1)
    return tck

def depth_to_spline_batch(depth_maps):
    """Process a batch of depth maps to generate spline values for each."""
    # Assuming depth_maps is a tensor of shape (batch_size, 128, 128, 4)
    # and the first channel of each depth map is used.
    batch_spline_values = []
    for depth_map in depth_maps:
        depth_channel = depth_map[:, :, 0]  # Extract the depth channel
        
        # Setup constants and meshgrid
        degree = 3
        n_control_points = 20
        x = np.linspace(-52e-3, 52e-3, 128)
        y = np.linspace(-52e-3, 52e-3, 128)
        X, Y = np.meshgrid(x, y)
        x_flat = X.flatten()
        y_flat = Y.flatten()

        # Flatten the depth channel and convert to numpy.
        z = tf.reshape(depth_channel, [-1]).numpy()

        # Create knots and fit spline
        tx = create_knots(degree, n_control_points, np.min(x_flat), np.max(x_flat))
        ty = create_knots(degree, n_control_points, np.min(y_flat), np.max(y_flat))
        tck = fit_Bspline(x_flat, y_flat, z, degree, tx, ty)

        # Calculate spline values and convert back to tensor
        bspline_values = bisplev(x, y, tck)
        batch_spline_values.append(bspline_values)

    # Convert list of arrays to a single tensor
    return tf.convert_to_tensor(batch_spline_values, dtype=tf.float32)

def spline_loss(depth_true, depth_pred):
    """Calculate the spline-based loss between true and predicted depth across a batch."""
    b_pred_values = depth_to_spline_batch(depth_pred)
    b_true_values = depth_to_spline_batch(depth_true)

    return tf.reduce_mean(tf.square(b_pred_values - b_true_values))

def combined_loss(y_true,y_pred):
    # print("y_true shape:", y_true.shape)
    # print("y_pred shape:", y_pred.shape)

    depth_true = y_true[:,:,:,0]
    normal_true = y_true[:,:,:,1:4] 

    depth_pred = y_pred[:,:,:,0]
    normal_pred = y_pred[:,:,:,1:4]

    depth_true = tf.expand_dims(depth_true, -1)
    depth_pred = tf.expand_dims(depth_pred, -1)

    alpha = 0.2
    beta = 0.2
    gamma = 0.2
    theta = 0.2
    delta = 0.2

    #depth loss
    loss_depth = alpha*(depth_loss(depth_true,depth_pred))

    #normal loss
    loss_normal = beta*(normal_loss(normal_true,normal_pred))
    
    #normal from depth
    normal_from_depth = depth_to_normal(depth_pred)
    loss_depth_to_normal = gamma*(normal_loss(normal_true,normal_from_depth)) 


    #scale_loss
    loss_scale = theta * scale_loss(depth_true, depth_pred, normal_true, normal_pred)
    # print("Depth_true shape outside: ", depth_true.shape)
    loss_splines = delta*spline_loss(depth_true, depth_pred)
    # print("CLEAR ON THE LOSS FUNCTIONS")

    return (loss_depth + loss_normal + loss_depth_to_normal + loss_scale + loss_splines)

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


    model = FluidNet()  # Adjust the number of classes if necessary
    model.compile(
        optimizer='adam', 
        loss=loss_funcs, 
        metrics=[
            root_mean_squared_error, 
            absolute_relative_error,
            accuracy_125,
            accuracy_15625,
            accuracy_1953125
        ],
        run_eagerly=True
    )
    model.summary()

    custom_objects = {
    'accuracy_125': accuracy_125,
    'accuracy_15625': accuracy_15625,
    'accuracy_1953125': accuracy_1953125,
    'combined_loss': combined_loss, 
    'absolute_relative_error': absolute_relative_error,
    'root_mean_squared_error': root_mean_squared_error
}

    return model , custom_objects


if __name__ == "__main__":

    model = create_model()

        
