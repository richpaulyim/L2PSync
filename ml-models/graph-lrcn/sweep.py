import numpy as np
import adjmattrain as adj

rates = (1/2.75)**np.arange(3,6)

# read partition data
perm = np.random.permutation(200000) 
with open('../deltaKURA.npy', 'rb') as f:
    datain = np.load(f, allow_pickle=True)[perm,:,:,:]
    dataout = np.load(f)[perm]
# slice training data 
X = datain[:160000]
y = dataout[:160000]
y = np.reshape(y, (-1,1))

# ======================================== # 
# training and saving models
# ======================================== #
vals = []
# for given number of frames shown
its = []
for batch in range(10, 12):
    for lr in rates:
        for i in range(1,24):
            # train and save the model
            training = X[:,:i,:,:]
            trainX = np.reshape(training, (y.shape[0],i,30,30,1))
            its.append(adj.trainandsave(trainX, y, 1, i,
                figurename=0, figure=False, nb_epoch=25,
                batch_size=2**batch, learning=rates[lr]))
        vals.append(its)

# save permutation
with open('accuracies.npy', 'wb') as f:
    np.save(f, vals)
    np.save(f, perm)
