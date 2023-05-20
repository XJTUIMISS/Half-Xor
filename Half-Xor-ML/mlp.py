import torch
import pickle
import torch.optim as optim
import time
import numpy as np
import os, math
import matplotlib.pyplot as plt
from dataset_gene import m
from network import Model, testModel

netdir = './network/' + str(m)
num_i = 32   #input layer nodes
num_h = 100   #hidden layer nodes
num_o = 1   #output layer nodes

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def plotFunction(filename, x, y, x2 = None, y2 = None, x3 = None, y3 = None):
    fig = plt.figure(num = 1,dpi = 120)
    ax = plt.subplot(111)
    isNone = lambda x:[float("inf"), float("inf")] if x is None else x

    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    
    ax.plot(x,y,label = "train",color ="blueviolet")
    if x2 is not None and y2 is not None:
        ax.plot(x2,y2,label = "test",color ="red")

    ax.set_xlabel(f"epoch(m={m})",color='black')
    ax.set_ylabel("RMSD",color='black')

    plt.legend()
    plt.show()
    plt.savefig(os.path.join("drawres",str(filename)))


 
class RMSD(torch.nn.Module):
 
    def __init__(self):
        super().__init__()
 
    def forward(self, outputs, labels):
        return torch.sqrt(torch.mean(torch.square((outputs - labels) / labels) , dtype=torch.float32))


if __name__ == "__main__":
    '''load dataset'''
    fr = open('train.pkl', 'rb')
    data_train = pickle.load(fr)
    data_train_target = pickle.load(fr)
    fr.close()
    dataset_train = torch.utils.data.TensorDataset(data_train, data_train_target)

    fr_test = open('test.pkl', 'rb')
    data_test = pickle.load(fr_test)
    data_test_target = pickle.load(fr_test)
    fr.close()
    dataset_test = torch.utils.data.TensorDataset(data_test, data_test_target)



    '''creat dataset loader and net'''
    data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size = 10000, shuffle = True)
    data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size = 10000, shuffle = True)

    if not os.path.exists(netdir): os.mkdir(netdir)
    if os.path.exists(os.path.join(netdir, 'net.pkl')): os.remove(os.path.join(netdir, 'net.pkl'))
    if os.path.exists(os.path.join(netdir, 'net_params.pkl')): os.remove(os.path.join(netdir, 'net_params.pkl'))
    if os.path.exists(os.path.join(netdir, 'net.pt')): os.remove(os.path.join(netdir, 'net.pt'))
    if os.path.exists(os.path.join(netdir, 'net_params.pt')): os.remove(os.path.join(netdir, 'net_params.pt'))
    
    
    epochs = 300
    print(f'epochs: {epochs}')
    model=Model(num_i,num_h,num_o)
    cost = torch.nn.MSELoss()

    cost_test = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    Loss_set_train = []
    Loss_set_test = []


    '''training'''
    criterion = RMSD()
    criterion_test = RMSD()
    for epoch in range(epochs):
        
        tag = True
        inputs, labels = None, None
        sum_loss_train = 0
        num_train = 0
        for data in data_loader_train:
            inputs, labels = data
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            sum_loss_train += loss
            num_train += 1
        

        sum_loss_test = 0
        num_test = 0
        iter_test = iter(data_loader_test)
        inputs_test, labels_test = None, None
        for data_test in iter_test:
            inputs_test, labels_test = data_test
            outputs_test = model(inputs_test)
            loss_test = torch.sqrt(cost(outputs_test / labels_test, labels_test / labels_test))
            sum_loss_test += loss_test
            num_test += 1
            
        print('epoch', epoch+1, 'finished', end = "      ")
        print('loss_test:', sum_loss_test.item() / num_test, end = "      ")
        print('loss_train:', sum_loss_train.item() / num_train)
        
        Loss_set_test.append(sum_loss_test.item() / num_test)
        Loss_set_train.append(sum_loss_train.item() / num_train)


        if tag: 
            sample_in = model(inputs_test)[0].item()
            sample_label = labels_test[0].item()
            print("actual: ", sample_label, "  estimate: ", sample_in," error: ", abs(sample_label - sample_in) / sample_label)
        

    model.eval()    # self.train(False).



    '''test'''
    testModel(model)


    '''save'''
    
    torch.save(model, os.path.join(netdir, 'net.pkl'))  # save entire net
    torch.save(model.state_dict(), os.path.join(netdir, 'net_params.pkl'))   # save parameter
    torch.save(model, os.path.join(netdir, 'net.pt'))  # save entire net
    torch.save(model.state_dict(), os.path.join(netdir, 'net_params.pt'))   # save parameter

    madrange = np.array([epoch for epoch in range(epochs)])
    plotFunction(f'loss-{m}.png', madrange, np.array(Loss_set_train), madrange, np.array(Loss_set_test))