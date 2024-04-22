import matplotlib.pyplot as plt
import tensorflow as tf

from model import FluidNet, create_model
from utils import combined_loss
from dataloader import load_and_preprocess_data



def train_model(model, input_tensors, depth_tensors, normal_tensors, epochs=10, batch_size=32):
    # Expand dimensions of depth tensor from [5400,128,128] to [5400,128,128,1] to match normal tensor
    depth_tensors_expanded = tf.expand_dims(depth_tensors, axis=-1)

    # Concatenate the depth and normal tensors along the last dimension to match the model's output
    combined_y_true = tf.concat([depth_tensors_expanded, normal_tensors], axis=-1)

    # Train the model with the combined ground truth tensor
    history = model.fit(input_tensors, combined_y_true,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.2)
    
    # Plot loss and accuracy
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(history.history['loss'], label='Train Loss')
    axs[0].plot(history.history['val_loss'], label='Validation Loss')
    axs[0].set_title('Loss')
    axs[0].legend()

    # Adjust this part if 'accuracy' is not the correct metric name
    axs[1].plot(history.history['accuracy'], label='Train Accuracy')
    axs[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[1].set_title('Accuracy')
    axs[1].legend()

    plt.savefig("plot/Loss_{}.png".format(epochs))   # save the figure to file
    plt.close()





if __name__ == "__main__":

    with tf.device('/cpu:0'):  # Use '/gpu:0' if TensorFlow-Metal is installed
        model = create_model()
        input_tensors, depth_tensors, normal_tensors = load_and_preprocess_data()
        train_model(model, input_tensors, depth_tensors, normal_tensors)

        

    


# TRAIN_NUM = 30600
# VAL_NUM = 5400
# BATCH_SIZE = 32




# def train():
#     # TRAIN
#     with tf.device("/gpu:0"):

#         model = FluidNet(nClasses = 1,
#                 nClasses1 = 3,  
#                 input_height = 128, 
#                 input_width  = 128)
#         model.summary()
        
#         # Load the preprocessed data 
#         gc.collect()
#         X_train = np.load(dir_references+"X_train{}.npy".format(TRAIN_NUM))
#         X_train = np.array(X_train)
#         print(X_train.shape)
#         y_train = np.load(dir_references+"Y_train{}.npy".format(TRAIN_NUM))
#         y_train = np.array(y_train)   
#         print(y_train.shape)

#         X_test = np.load(dir_references+"X_val{}.npy".format(VAL_NUM))
#         X_test = np.array(X_test)
#         print(X_test.shape)
#         y_test = np.load(dir_references+"Y_val{}.npy".format(VAL_NUM))
#         y_test = np.array(y_test)   
#         print(y_test.shape)

#         #create model and train
#         training_log = TensorBoard(log_folder)
#         weight_filename = weight_folder + "pretrained_FSRN_CNN.h5"

#         stopping = EarlyStopping(monitor='val_loss', patience=2)

#         checkpoint = ModelCheckpoint(weight_filename,
#                                     monitor = "val_loss",
#                                     save_best_only = True,
#                                     save_weights_only = True)
#         #Plot loss
#         dir_plot = "plot/" 
        
#         model = FluidNet(nClasses     = 1,
#                 nClasses1 = 3,  
#                 input_height = 128, 
#                 input_width  = 128)
        
#         model.summary()
#         plot_model(model,to_file=dir_plot+'model.png',show_shapes=True)
        
#         epochs = 35
#         learning_rate = 0.001
#         batch_size = BATCH_SIZE

#         loss_funcs = {
#             "single_out": combined_loss,
#         }
#         loss_weights = {"single_out": 1.0}

        
#         print('lr:{},epoch:{},batch_size:{}'.format(learning_rate,epochs,batch_size))
        
#         adam = Adam(lr=learning_rate)
#         model.compile(loss= loss_funcs,
#                 optimizer=adam,
#                 loss_weights=loss_weights,
#                 metrics=["accuracy"])  
#         gc.collect()
        
#         history = model.fit(X_train,y_train,
#                         batch_size=batch_size,
#                         epochs=epochs,
#                         verbose=2, 
#                         validation_data=(X_test,y_test),
#                         callbacks = [training_log,checkpoint])
        
#         fig = plt.figure(figsize=(10,20)) 

#         ax = fig.add_subplot(1,2,1)
#         for key in ['loss', 'val_loss']:
#             ax.plot(history.history[key],label=key)
#         ax.legend()

#         ax = fig.add_subplot(1,2,2)
#         for key in ['acc', 'val_acc']:
#             ax.plot(history.history[key],label=key)
#         ax.legend()
#         fig.savefig(dir_plot+"Loss_"+str(epochs)+".png")   # save the figure to file
#         plt.close(fig)

