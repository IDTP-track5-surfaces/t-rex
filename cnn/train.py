import numpy as np
import matplotlib.pyplot as plt
import gc
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import plot_model
import tensorflow as tf

from cnn.model import FluidNet
from cnn.utils import combined_loss


TRAIN_NUM = 30600
VAL_NUM = 5400
BATCH_SIZE = 32
dir_references = "data/"

def train():
    # TRAIN
    with tf.device("/gpu:0"):

        model = FluidNet(nClasses = 1,
                nClasses1 = 3,  
                input_height = 128, 
                input_width  = 128)
        model.summary()
        
        # Load the preprocessed data 
        gc.collect()
        X_train = np.load(dir_references+"X_train{}.npy".format(TRAIN_NUM))
        X_train = np.array(X_train)
        print(X_train.shape)
        y_train = np.load(dir_references+"Y_train{}.npy".format(TRAIN_NUM))
        y_train = np.array(y_train)   
        print(y_train.shape)

        X_test = np.load(dir_references+"X_val{}.npy".format(VAL_NUM))
        X_test = np.array(X_test)
        print(X_test.shape)
        y_test = np.load(dir_references+"Y_val{}.npy".format(VAL_NUM))
        y_test = np.array(y_test)   
        print(y_test.shape)

        #create model and train
        training_log = TensorBoard(log_folder)
        weight_filename = weight_folder + "pretrained_FSRN_CNN.h5"

        stopping = EarlyStopping(monitor='val_loss', patience=2)

        checkpoint = ModelCheckpoint(weight_filename,
                                    monitor = "val_loss",
                                    save_best_only = True,
                                    save_weights_only = True)
        #Plot loss
        dir_plot = "plot/" 
        
        model = FluidNet(nClasses     = 1,
                nClasses1 = 3,  
                input_height = 128, 
                input_width  = 128)
        
        model.summary()
        plot_model(model,to_file=dir_plot+'model.png',show_shapes=True)
        
        epochs = 35
        learning_rate = 0.001
        batch_size = BATCH_SIZE

        loss_funcs = {
            "single_out": combined_loss,
        }
        loss_weights = {"single_out": 1.0}

        
        print('lr:{},epoch:{},batch_size:{}'.format(learning_rate,epochs,batch_size))
        
        adam = Adam(lr=learning_rate)
        model.compile(loss= loss_funcs,
                optimizer=adam,
                loss_weights=loss_weights,
                metrics=["accuracy"])  
        gc.collect()
        
        history = model.fit(X_train,y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=2, 
                        validation_data=(X_test,y_test),
                        callbacks = [training_log,checkpoint])
        
        fig = plt.figure(figsize=(10,20)) 

        ax = fig.add_subplot(1,2,1)
        for key in ['loss', 'val_loss']:
            ax.plot(history.history[key],label=key)
        ax.legend()

        ax = fig.add_subplot(1,2,2)
        for key in ['acc', 'val_acc']:
            ax.plot(history.history[key],label=key)
        ax.legend()
        fig.savefig(dir_plot+"Loss_"+str(epochs)+".png")   # save the figure to file
        plt.close(fig)

