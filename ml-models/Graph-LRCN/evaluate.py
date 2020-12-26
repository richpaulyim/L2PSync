import tensorflow as tf
import numpy as np

omega = tf.keras.models.load_model('models/Omega')

perm = np.random.permutation(200000)
seq_length=15
# Make data selection and sequence length
with open('../LRCN-Data/Differential.npy', 'rb') as f:
    datain = np.load(f, allow_pickle=True)[perm,:seq_length,:,:]
    dataout = np.load(f)[perm]
    # data
    X_val = datain[160000:]
    X_val = np.reshape(X_val, (40000,seq_length,15,15,1))
    y_val = dataout[160000:]
    y_val = np.reshape(y_val, (-1,1))
    results = diff.evaluate(X_val, y_val, batch_size=128)
    print("test loss, test acc:", results)
