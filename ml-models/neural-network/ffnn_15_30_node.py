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

its = [390 - (15*i) for i in range(25)]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('using gpu')

data = np.load('sampled_kura15new.npy', allow_pickle=True)



df1 = pd.read_csv('sync (52).csv', header=None)
y = df1.to_numpy()

for it in tqdm(its):
#hyperparameters
    input_size = 400 - it
    hidden_size = 100
    num_classes = 2
    num_epochs = 25
    batch_size = 256 
    learning_rate = 0.01
    LAMBDA_1 = 0.01
    LAMBDA_2 = 0.01 
    decay = 2e-5

    print('Using first ' + str((390 - it)/15)  + ' iterations')
    
    new_data = []
    print('reformatting data')
    for i in trange(data.shape[0]):
        a = np.delete(data[i], list(range(390 - it, 390)))
        new_data.append(a)
    new_data = np.array(new_data)
    
    np.reshape(new_data, newshape=(50000, input_size))
  
    print('assigning data')

    train_data, test_data, train_labels, test_labels = train_test_split(new_data, y, test_size=0.10, random_state=57109, shuffle=True)
    train_labels = np.ravel(train_labels)
    test_labels = np.ravel(test_labels)

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
            
            out = net(test_datat.float())
            _, predictedt = torch.max(out.data, 1)
            test_score = len(predictedt[predictedt == test_labelst]) / len(test_labelst)
            
            final_score = test_score
            loss =  criterion(outputs, train_labelst[i: i + batch_size])
      
            
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
            
        
            
    print('=======================================================')

    #testing phase
    print(final_scores)

    #testing on test data
    out = net(test_datat.float())
    _, predicted = torch.max(out.data, 1)
    test_score = len(predicted[predicted == test_labelst]) / len(test_labelst)
    print(test_score)
 
    #computing points of interest
    k = len(predicted[predicted==1])
    n = len(predicted[predicted==0])
    score = final_scores[num_epochs-1][0]
    ITS_SCORE.append(float(score))
    #printing out points of interest to console
    print('Amount of zeros predicted is: ', str(n))
    print('Amount of zeros there actually are: ', str(len(np.where(test_labels == 0)[0])))      
    print('Amount of ones predicted is: ', str(k))
    print('Amount of ones there actually are: ', str(len(np.where(test_labels == 1)[0])))
    print('Accuracy is :' , str(score))

    #plotting results
    plt.plot(test_acc)
    plt.plot(train_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('FFNN Accuracy Over Time')
    plt.legend(["Testing Accuracy", "Training Accuracy"])
    plt.savefig('Accuracy_' + str(it) + 'its_200k_10node')
    #plt.show()

    #plotting losses
    plt.clf()
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('FFNN Loss Over Epochs')
    plt.savefig('Loss_' + str(it) + 'its_200k_10node')
    #plt.show()


    print("Iterations: " + str(i)+'\n'+'Time: ' + \
                          time_format(round(time.time()-s, 3)) + '\n')
print(ITS_SCORE)
with open('kura15rerunFFNN.npy', 'wb') as npyf:
    #save numpy binary file 
    np.save(npyf, np.array(ITS_SCORE))