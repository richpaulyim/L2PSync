import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys; sys.path.insert(1, '../../')
from firefly import time_format
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from tqdm import trange, tqdm

# start time
s = time.time()
ITS_SCORE = []

its = [240, 230, 220, 210, 200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('using gpu')
#hyperparameters
hidden_size = 100
tau=10
num_classes = 2
num_epochs = 2
batch_size = 128
learning_rate = 0.0001
LAMBDA_1 = 0.01
LAMBDA_2 = 0.01 
decay = 2e-5
fpr = []
tpr = []
input_size = 10 
input_size = 730
a = 4
for i in np.asarray([1,2,4,8,12,16]):
    subgraphs = int(i)
    # read in data
    if a==0:
        data = pd.read_csv('features_SG20FCA25K.csv', header=None).to_numpy()
        df = pd.read_csv('sync_SG20FCA25K.csv', header=None).to_numpy()
        subgraphrange = 20
    if a==1:
        data = pd.read_csv('~/Downloads/features_fca16kvaried_subgraphs.csv', header=None).to_numpy()
        df = pd.read_csv('~/Downloads/sync_fca16kvaried_subgraphs.csv', header=None).to_numpy()
        subgraphrange = 16
    if a==2:
        data = pd.read_csv('subgraphs_varied300to600_features.csv', header=None).to_numpy()
        df = pd.read_csv('subgraphs_varied300to600_sync.csv', header=None).to_numpy()
        subgraphrange = 16
    if a==3:
        print("READING IN SYNC")
        df = pd.read_csv('/mnt/l/home/fixed600_dat/fcaFIXED600_subgraphs_sync.csv', header=None).to_numpy()
        print("READING IN FEATURES")
        data = pd.read_csv('/mnt/l/home/fixed600_dat/fcaFIXED600_subgraphs_features.csv', header=None).to_numpy()
    if a==4:
        print("READING IN SYNC")
        data = pd.read_csv('/mnt/l/home/fcaFIXEDNODES_VARIEDDENSITY/fcaEDGEVARIED600_subgraphs_features.csv', header=None).to_numpy()
        print("READING IN FEATURES")
        df = pd.read_csv('/mnt/l/home/fcaFIXEDNODES_VARIEDDENSITY/fcaEDGEVARIED600_subgraphs_sync.csv', header=None).to_numpy()
        subgraphrange = 16
    print(data.shape, df.shape)
    y = np.asarray([[int(k[0][1])] for k in df])
    parenty = y[np.arange(len(y),step=subgraphrange)]
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    indsync = np.random.permutation(np.where(parenty==1)[0])[1:]
    indnosync = np.random.permutation(np.where(parenty==0)[0])[1:]
    it=24
    # split-by-hand and indices shuffled and separated
    slicept = 12800
    trainind = [np.arange(subgraphs)+i*subgraphrange for i in indsync[:slicept]] +\
        [np.arange(subgraphs)+i*subgraphrange for i in indnosync[:slicept]]
    testind = [np.arange(subgraphs)+i*subgraphrange for i in indsync[slicept:16000]] +\
        [np.arange(subgraphs)+i*subgraphrange for i in indnosync[slicept:16000]]
    trainind = np.random.permutation(np.asarray([x for t in trainind for x in t]))
    testind = np.asarray([x for t in testind for x in t])
    print("Number of subgraphs observed")
    print(subgraphs)
    # ===============================
    # slicing the data
    # ===============================
    #columns = np.append(np.arange(24),np.arange(720,730))
    #train_data = data[trainind,columns[:,np.newaxis]]
    #test_data = data[testind,columns[:,np.newaxis]]
    if input_size == 10:
        train_data = data[trainind,-10:]
        test_data = data[testind,-10:]
    if input_size == 730:
        train_data = data[trainind,:]
        test_data = data[testind,:]
    train_labels = np.ravel(y[trainind])
    test_labels = np.ravel(y[testind])

    # data2 = np.load('C:/Users/hbass/Desktop/fca/FCA-ML/machine-learning/neural-network/features (4).npy', allow_pickle=True)
    # df2 = pd.read_csv('C:/Users/hbass/Desktop/fca/FCA-ML/machine-learning/neural-network/sync (8).csv', header=None)
    # y2 = df2.to_numpy()

    # data = np.vstack((data1, data2))
    # y = np.vstack((y1, y2))

    #y = np.load('C:/Users/hbass/Desktop/fca/FCA-ML/machine-learning/neural-network/labels (2).npy')
    # else:
    #     data = np.load('/home/richpaulyim/Projects/FCA-ML/data-generation/sampling_data/sampling_data.npy', allow_pickle=True)
    #     y = np.load('/home/richpaulyim/Projects/FCA-ML/data-generation/sampling_data/sampling_data_labels.npy')


    #FFNN class
    class Net(nn.Module):
     
        def __init__(self, input_size, hidden_size, num_classes):
            super(Net, self).__init__()                    

            # layer 1
            self.fc1 = nn.Linear(input_size, 100) 
            self.batch1 = nn.BatchNorm1d(100)
            self.drop1 = nn.Dropout(0.25)

            # layer 2
            self.fc2 = nn.Linear(100, 80)
            self.batch2 = nn.BatchNorm1d(80)
            self.drop2 = nn.Dropout(0.25)

            # layer 3
            self.fc3 = nn.Linear(80, 60)
            self.batch3 = nn.BatchNorm1d(60)
            self.drop3 = nn.Dropout(0.25)

            # layer 4
            self.fc4 = nn.Linear(60, 40)
            self.batch4 = nn.BatchNorm1d(40)
            self.drop4 = nn.Dropout(0.25)

            self.fc5 = nn.Linear(40, num_classes)

            self.relu = nn.Sigmoid() 

        def forward(self, x):                            
          
            # layer 1
            out = self.fc1(x)
            out = self.batch1(out)
            out = self.relu(out)
            out = self.drop1(out)

            # layer 2
            out = self.fc2(out)
            out = self.batch2(out)
            out = self.relu(out)
            out = self.drop2(out)

            # layer 3
            out = self.fc3(out)
            out = self.batch3(out)
            out = self.relu(out)
            out = self.drop3(out)

            # layer 4
            out = self.fc4(out)
            out = self.batch4(out)
            out = self.relu(out)
            out = self.drop4(out)

            out = self.fc5(out)

            return out

    #setting up and converting for pytorch neural network
    final_scores = []
    test_acc = []
    train_acc = []
    losses = []
    final_test = 0

    train_datat = torch.from_numpy(train_data)
    #print(train_labels)
    #print(test_labels)
    train_labelst = torch.from_numpy(train_labels)

    test_datat = torch.from_numpy(test_data)
    test_labelst = torch.from_numpy(test_labels)

    print('TESTING: ' +  'learning_rate = ' + str(learning_rate))

    net = Net(input_size, hidden_size, num_classes)
    net = net.to(device)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_datat = train_datat.to(device)
    train_labelst = train_labelst.to(device)
    test_datat = test_datat.to(device)
    test_labelst = test_labelst.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    criterion1 = nn.L1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    #training the FFNN
    for epoch in range(1, num_epochs + 1):
        final_score = 0
        final_loss = 0
        for i in range(0, train_datat.shape[0], batch_size):
          
            optimizer.zero_grad()

            if i + batch_size > train_datat.shape[0]:
                batched_data = train_datat[i:]
            else:
                batched_data = train_datat[i: i + batch_size]
            outputs = net(batched_data.float())
         
            _, predicted = torch.max(outputs.data, 1)
            train_score = len(predicted[predicted == train_labelst[i: i + batch_size]]) / len(train_labelst[i: i + batch_size])
          
            #out = net(test_datat.float())
            #_, predictedt = torch.max(out.data, 1)
            #test_score = len(predictedt[predictedt == test_labelst]) / len(test_labelst)
          
            #final_score = test_score
            final_score=0
            test_score=0
            loss =  criterion(outputs, train_labelst[i: i + batch_size])
            # for param in net.parameters():
            #     loss +=  (LAMBDA_2 * (torch.norm(param, 2) ** 2))
            #     #loss += LAMBDA_2 * ((1-LAMBDA_1) * torch.norm(param, 2) ** 2)
          
            final_loss = loss
          
            loss.backward()
            optimizer.step()
          
          
      
        info = {0: final_score, 1: learning_rate, 2: hidden_size}
      
        final_scores.append(info)
        losses.append(loss)
        train_acc.append(train_score)
        test_acc.append(test_score)
      
        print('Epoch: {}/{}.............'.format(epoch, num_epochs), end=' ') 
        print("Loss: {:.4f}".format(loss.item()))
          

    #testing phase
    #print(final_scores)

    #testing on test data
    out = net(test_datat.float())
    _, predicted = torch.max(out.data, 1)
    out.data.cpu()
    test_score = len(predicted[predicted == test_labelst]) / len(test_labelst)
    torch.set_printoptions(threshold=10000)
    # ========================= #
    #    MAJORITY CLASSIFIER 
    # ========================= #
    subpred = np.asarray((predicted==test_labelst).cpu())
    subgraphclasses = [subpred[np.arange(subgraphs)+i*subgraphs] for i in
            range(int(len(subpred)/subgraphs))] 
    finalpred = np.sum(np.asarray(subgraphclasses),axis=1)
    import matplotlib.pyplot as plt
    from sklearn import metrics
    pred = predicted.cpu().numpy()
    y_true = test_labels[np.arange(len(test_labels),step=subgraphs)] # ground truth labels
    groupedpred = np.asarray([sum(pred[np.arange(subgraphs) + i*subgraphs]) for i in
        np.arange(int(len(pred)/subgraphs))])
    print("groupedpred length:", len(groupedpred))
    scores = groupedpred/subgraphs # predicted probabilities generated by sklearn classifier
    fprtemp, tprtemp, thresholds = metrics.roc_curve(y_true+1, scores, pos_label=2)
    fpr.append(fprtemp)
    tpr.append(tprtemp)
    print('=============================================')
    print('CHILDREN ACCURACY')
    print(metrics.confusion_matrix(predicted.cpu(),test_labels))
    print("PARENT ACCURACY") 
    print(metrics.confusion_matrix(scores>0.5,
        test_labels[np.arange(len(test_labels),step=subgraphs)]))
    print('=============================================')
    torch.cuda.empty_cache()

# save the ture positive and false positive rates
with open("roc" +"_"+str(a)+"_"+str(input_size)+"_varied.npy", "wb") as f:
    np.save(f,fpr)
    np.save(f,tpr)

# ========================= #
#           ROC 
# ========================= #
plt.figure()
for i in range(len(fpr)):
    plt.plot(fpr[i],tpr[i],lw=2,label="AUC: %0.3f" % metrics.auc(fpr[i],tpr[i]))
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title("ROC curve "+str(input_size)+str(a))
plt.legend(loc='lower right')
plt.show()
