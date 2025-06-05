# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:35:50 2023

@author: arwilzman
"""
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn import preprocessing
import numpy as np
import time


#%%
def train_supervised(training_inputs, training_outputs, network, epochs, learning_rate, 
          wtdecay,batch_size, loss_function, print_interval): 
  # convert numpy data to tensor data for pytorch
  train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(training_inputs),
                                                 torch.FloatTensor(training_outputs))
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                             shuffle=True)
  optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=wtdecay)
  track_losses = np.zeros(epochs)
  start = time.time()
  for epoch in range(1, epochs+1):
      for batch_idx, (X, y) in enumerate(train_loader):
          # grab the x's for this batch
          data = X
          # find the predictions f(x) for this batch
          output = network(data.to(device))
          # find the loss
          loss = loss_function(output, y.to(device))
          # compute the gradient and update the network parameters
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
      # housekeeping - keep track of our losses and print them as we go
      training_loss = loss.item()**.5
      track_losses[epoch-1] = training_loss
      if epoch % print_interval == 0:
          print('epoch: %4d training loss:%10.3e time:%7.1f'%(epoch, training_loss, time.time()-start)) 
  return network, track_losses

#%%
def train_supervised_hyper(batch_size,num_epochs,learn_rate,seed,x_train,x_test,y_train,y_test):
    #gc.collect()
    #torch.cuda.empty_cache()
    sz = np.size(x_train,1)
    class testnet(nn.Module):
        def __init__(self):
            super(testnet, self).__init__()
            self.structure = nn.Sequential(
                nn.Linear(sz,sz),
                nn.LeakyReLU(),
                nn.Linear(sz,5),
                nn.LeakyReLU(),
                nn.Linear(5,1)
            )
        def forward(self, x):
            x = self.structure(x)
            return x
    torch.manual_seed(seed)
    model_1 = testnet().cuda()
    crit = nn.L1Loss()
    p_int = 1000
    testnet, losses = train(x_train, y_train, model_1, num_epochs, learn_rate, batch_size, crit, p_int)
    plt.plot(losses)
    plt.title('Test: ' + str(batch_size) + ' batch,' + str(learn_rate) + ' learn, ' + 
              'seed: ' +str(seed))
    plt.ylim((0,.5))
    plt.show()
    [MAE_train_nn1,MSE_train_nn1,RMSE_train_nn1,R2_train_nn1,
     MAE_test_nn1,MSE_test_nn1,RMSE_test_nn1,R2_test_nn1]=model_eva(x_train,x_test,y_train,y_test,testnet)
    return [MAE_train_nn1,MSE_train_nn1,RMSE_train_nn1,R2_train_nn1,MAE_test_nn1,MSE_test_nn1,RMSE_test_nn1,R2_test_nn1]

#%%
def train_unsupervised(training_inputs, network, epochs, learning_rate, 
          wtdecay, batch_size, loss_function, print_interval): 
  # convert numpy data to tensor data for pytorch
  train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(training_inputs),
                                                 torch.FloatTensor(training_inputs))
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                             shuffle=True)
  optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=wtdecay)
  track_losses = np.zeros(epochs)
  start = time.time()
  for epoch in range(1, epochs+1):
      for batch_idx, (X, y) in enumerate(train_loader):
          # grab the x's for this batch
          data = X
          # find the predictions f(x) for this batch
          output = network(data.to(device))
          # find the loss
          loss = loss_function(output, data.to(device))
          # compute the gradient and update the network parameters
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
      # housekeeping - keep track of our losses and print them as we go
      training_loss = loss.item()**.5
      track_losses[epoch-1] = training_loss
      if epoch % print_interval == 0:
          print('epoch: %4d training loss:%10.3e time:%7.1f'%(epoch, training_loss, time.time()-start)) 
  return network, track_losses

#%%
def model_eval_supervised(XTrain,XTest,yTrain,yTest,model):
    XTrain_th=torch.FloatTensor(XTrain).cuda()
    XTest_th=torch.FloatTensor(XTest).cuda()
    yTrain_pred_np=model(XTrain_th).cpu().detach().numpy()
    yTest_pred_np=model(XTest_th).cpu().detach().numpy()
    MAE_train = np.zeros((yTrain.shape[0],1))
    MSE_train = np.zeros((yTrain.shape[0],1))
    RMSE_train = np.zeros((yTrain.shape[0],1))
    R2_train = np.zeros((yTrain.shape[0],1))
    MAE_test = np.zeros((yTrain.shape[0],1))
    MSE_test = np.zeros((yTrain.shape[0],1))
    RMSE_test = np.zeros((yTrain.shape[0],1))
    R2_test = np.zeros((yTrain.shape[0],1))
    for i in range(yTrain.shape[0]):
        MAE_train[i,0] = mean_absolute_error(yTrain_pred_np[i,:],yTrain[i,:])
        MSE_train[i,0] = mean_squared_error(yTrain_pred_np[i,:],yTrain[i,:])
        RMSE_train[i,0] = np.sqrt(mean_squared_error(yTrain_pred_np[i,:],yTrain[i,:]))
        R2_train[i,0] = r2_score(yTrain_pred_np[i,:],yTrain[i,:])
    for i in range(yTest.shape[0]):
        MAE_test[i,0] = mean_absolute_error(yTest_pred_np[i,:],yTest[i,:])
        MSE_test[i,0] = mean_squared_error(yTest_pred_np[i,:],yTest[i,:])
        RMSE_test[i,0] = np.sqrt(mean_squared_error(yTest_pred_np[i,:],yTest[i,:]))
        R2_test[i,0] = r2_score(yTest_pred_np[i,:],yTest[i,:]) 
    plt.scatter(yTrain_pred_np.flatten(),yTrain.flatten())
    plt.title('Train')
    plt.xlabel('predicted value')
    plt.ylabel('actual value')
    MAE_tr = np.mean(MAE_train)
    MSE_tr = MSE_train.mean()
    RMSE_tr = RMSE_train.mean()
    R2_tr = R2_train.mean()
    MAE_te = MAE_test.mean()
    MSE_te = MSE_test.mean()
    RMSE_te = RMSE_test.mean()
    R2_te = R2_test.mean()
    train_p = max(yTrain_pred_np.flatten())+.1*max(yTrain_pred_np.flatten())
    train_a = max(yTrain.flatten())
    test_p = max(yTest_pred_np.flatten())+.1*max(yTest_pred_np.flatten())
    test_a = max(yTest.flatten())
    plt.text(train_p,train_a,('MAE = '+str(round(MAE_tr,2))))
    plt.text(train_p,train_a-(.1*train_a),('MSE = '+str(round(MSE_tr,2))))
    plt.text(train_p,train_a-(.2*train_a),('RMSE = '+str(round(RMSE_tr,2))))
    plt.text(train_p,train_a-(.3*train_a),('R2 = '+str(round(R2_tr,2))))
    plt.show()
    plt.scatter(yTest_pred_np.flatten(),yTest.flatten())
    plt.title('Test')
    plt.xlabel('predicted value')
    plt.ylabel('actual value')
    plt.text(test_p,test_a,('MAE = '+str(round(MAE_te,2))))
    plt.text(test_p,test_a-(.1*test_a),('MSE = '+str(round(MSE_te,2))))
    plt.text(test_p,test_a-(.2*test_a),('RMSE = '+str(round(RMSE_te,2))))
    plt.text(test_p,test_a-(.3*test_a),('R2 = '+str(round(R2_te,2))))
    plt.show()
    return MAE_tr,MSE_tr,RMSE_tr,R2_tr,MAE_te,MSE_te,RMSE_te,R2_te

#%%
def model_eval_unsupervised(x,model):
  a = torch.FloatTensor(x).cuda()
  b = model.encode(a)
  c = model.decode(b)
  x = a.cpu().detach().numpy().reshape(-1)
  enx = b.cpu().detach().numpy().reshape(-1)
  dex = c.cpu().detach().numpy().reshape(-1)
  MAE_train=mean_absolute_error(x,dex)
  MSE_train=mean_squared_error(x,dex)
  RMSE_train=np.sqrt(mean_squared_error(x,dex))
  R2_train=r2_score(x,dex)
  return MAE_train,MSE_train,RMSE_train,R2_train

#%%
def plot_network(title,network,data,pred,losses,MAE,RMSE,R2):
    plt.figure()
    ax1 = plt.subplot(231)
    ax2 = plt.subplot(234)
    ax3 = plt.subplot(132)
    ax4 = plt.subplot(233)
    ax5 = plt.subplot(236)
    
    ax1.plot(losses)
    ax1.set_title(title)
    ax2.bar('MAE',MAE,color='red')
    ax2.bar('RMSE',RMSE,color='blue')
    ax3.bar('R2',R2)
    y_max = max(losses)
    ax1.set_ylim([0,y_max])
    ax2.set_ylim([0,y_max])
    ax3.set_ylim([0,1])
    plt.show()
    predavg = np.average(pred,axis=0)
    dataavg = np.average(data,axis=0)
    AE = abs(predavg - dataavg)
    ax4.plot(AE)
    ax4.set_ylabel('Absolute Error')
    ax4.text(0,min(AE),'feature #')
    ax4.text(0,max(AE)-(.1*max(AE)),('MAE = ' + str(round(np.mean(AE),2))))
    ax4.text(3,max(AE)-(.1*max(AE)),('R2 = ' + str(round(R2,2))))
    ax5.scatter(pred.flatten(),data.flatten())
    ax5.set_ylabel('actual data')
    ax5.set_xlabel('decoded data')
    ax4.set_title(title)
    plt.show()