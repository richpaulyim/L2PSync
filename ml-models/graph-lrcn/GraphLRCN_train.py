"""
Train our RNN on extracted features or images.
"""
import numpy as np 
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger 
import keras
from identity import fca_true_classify
from sklearn import metrics
from GraphLRCN import GraphLSTM
import time, sys
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
import keras.models
np.set_printoptions(threshold=sys.maxsize)

def train_model(datain, dataout, seq_length, model, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=True, batch_size=32, nb_epoch=100, lrate=0.001):

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('data', 'logs', model))

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    # steps_per_epoch = (len(data.data) * 0.7) // batch_size

    # Get the model.
    rm = GraphLSTM(2, seq_length, saved_model, model, lrate=lrate)

    # Fit!
    if load_to_memory:
        # Use standard fit.
        return (rm.model.fit(
            datain,
            dataout,
            batch_size=batch_size,
            validation_split=0.3,
            verbose=1,
            callbacks=[tb],
            epochs=nb_epoch),
            rm.model)


def trainandsave(datain, dataout, selection, seq_length, figurename,
        figure=True, nb_epoch=100, batch_size=512, learning=0.001):
    """These are the main training settings. Set each before running
    this file."""
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d
    model = 'lrcn'
    saved_model = None  # None or weights file
    class_limit = 1 # int, can be 1-101 or None

    # Chose images or features and image shape based on network.
    if model=='lrcn':
        image_shape = (15, 15, 1)
    else:
        raise ValueError("Invalid model. See train.py for options.")
   
    # Make data selection and sequence length
    if selection==0:
        selection = 1
    
    if selection==1:
        print('Data was read.')
    if selection==2:
        with open('../LRCN-Data/LineConv.npy', 'rb') as f:
            datain = np.load(f, allow_pickle=True)[perm,:seq_length,:,:]
            dataout = np.load(f)[perm]
    if selection==3:
        with open('../LRCN-Data/LoUp.npy', 'rb') as f:
            datain = np.load(f, allow_pickle=True)[perm,:seq_length,:,:]
            dataout = np.load(f)[perm]
    if selection==4:
        with open('../LRCN-Data/delta.npy', 'rb') as f:
            datain = np.load(f, allow_pickle=True)[perm,1:(seq_length+1),:,:]
            dataout = np.load(f)[perm]

    hs, model_eval = train_model(datain, dataout, seq_length, model, 
            saved_model=saved_model,
            class_limit=class_limit, image_shape=image_shape,
          batch_size=batch_size, nb_epoch=nb_epoch, lrate=learning)

    # save model
    if figurename: model_eval.save('models/'+figurename)

    # --------------------------------------
    # training loss and accuracy study
    # --------------------------------------
    # loss study, save loss figure
    acc = hs.history['accuracy']
    val_acc = hs.history['val_accuracy']
    if figure:
        loss = hs.history['loss']
        val_loss = hs.history['val_loss']
        epochs = np.arange(len(loss)) + 1
        tl, = plt.plot(epochs, loss, 'bo', label='Training loss')
        vl, = plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.legend(handles=[tl,vl], loc='upper right')
        plt.title('Loss ('+figurename+')')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('figures/'+figurename+'_accuracy.png')
        plt.clf()

        # accuracy study, save loss figure
        ta, = plt.plot(epochs, acc, 'bo', label='Training acc')
        va, = plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.legend(handles=[ta,va], loc='lower right')
        plt.title('Accuracy ('+figurename+')') 
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig('figures/'+figurename+'_loss.png')
        plt.close()
    return val_acc

if __name__ == '__main__':

    train = True 
    test = False 

    # train model
    if train:
        # read partition data
        perm = np.random.permutation(160000) 
        with open('../DeltaKURA.npy', 'rb') as f:
            #datain = np.load(f, allow_pickle=True)[perm,:23,:,:]
            datain = np.load(f, allow_pickle=True)
            print(datain.shape)
            import pdb;pdb.set_trace()
            dataout = np.load(f)[perm]

        # slice training data 
        X = datain[:160000]
        y = dataout[:160000]
        y = np.reshape(y, (-1,1))

        # save validation accuracies
        vals = []
        for i in range(4,5):
            # train and save the model
            training = X[:,:i,:,:]
            trainX = np.reshape(training, (y.shape[0],i,15,15,1))
            vals.append(trainandsave(trainX, y, 1, i, 'DeltaKURA', 3, False))

        # save results
        with open('differentialKURALRCN_addon.npy', 'wb') as f:
            np.save(f,np.asarray(vals))

        # save permutation
        with open('models/testindicesKURA.npy', 'wb') as f:
            np.save(f, perm)

    # test trained model
    if test:
        # load saved model
        reconstructed_model = keras.models.load_model("models/DifferentialGH")
        seq_length = 9
        
        # load permutation array
        with open('models/testindicesGH.npy', 'rb') as f:
            permutation = np.load(f, allow_pickle=True)
        # load and permuate data
        with open('../DifferentialGH.npy', 'rb') as f:
            datain = np.load(f, allow_pickle=True)[permutation,:seq_length,:,:]
            dataout = np.load(f)[permutation]
        
        # slice testing data
        X_test = datain[160000:]
        y_test = dataout[160000:]
        print("------------------------------")
        print(len(y_test))
        print("------------------------------")

        # classify data, create indices
        identity_ind = []
        for i, obj in enumerate(X_test):
            identity_ind.append(fca_true_classify(obj, 9))
        print("------------------------------")
        print("Correctly classified ", np.sum(identity_ind))
        print("------------------------------")
        identity_ind = np.logical_not(identity_ind)

        # reshape remaining unclassified data
        X_test = np.reshape(X_test[identity_ind,:,:,:], 
                (np.sum(identity_ind),seq_length,15,15,1))
        y_test = y_test[identity_ind]
        
        # evaluate on loaded model
        labels = reconstructed_model.predict_classes(X_test)
        print("------------------------------")
        print(metrics.confusion_matrix(y_test,labels))
        print(np.mean(y_test==labels))
        print("------------------------------")

        # old function calls
        #trainandsave(4, 9, 'Delta')
        #trainandsave(3, 10, 'LowerUpper')
        #trainandsave(2, 10, 'LineConv')

