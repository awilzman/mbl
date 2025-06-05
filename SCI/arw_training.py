# -*- coding: utf-8 -*-
# =============================================================================
# author: Andrew R. Wilzman
# functions:
# mask_registration(file,target_res,i,drop_factor)
#   purpose: grab dicom data, return indices and densities from point cloud
# train_supervised(training_inputs, training_outputs, network, epochs, learning_rate, 
#          wtdecay,batch_size, loss_function, print_interval, device):
#    purpose: train a supervised model    
# train_unsupervised(training_inputs, network, epochs, learning_rate, 
#          wtdecay, batch_size, loss_function, print_interval,device)
#    purpose: train an unsupervised model    
# model_eval_supervised(XTrain,XTest,yTrain,yTest,model)
#    purpose: evaluate supervised model
# model_eval_unsupervised(x,model)
#    purpose: evaluate unsupervised model 
#    plot_network(title,network,data,pred,losses,MAE,RMSE,R2)
#    purpose: plot network metrics
# =============================================================================
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
          wtdecay,batch_size, loss_function, print_interval, device): 
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

def train_unsupervised(training_inputs, network, epochs, learning_rate, 
          wtdecay, batch_size, loss_function, print_interval,device): 
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
    #ax1.set_ylim([0,y_max])
    #ax2.set_ylim([0,y_max])
    #ax3.set_ylim([0,1])
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
    
# GAN training function
def train_GD(training_inputs, Gnet, Dnet, epochs, learning_rate, 
          batch_size, loss_function, print_interval): 
  # convert numpy data to tensor data for pytorch
  train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(training_inputs),
                                                 torch.FloatTensor(training_inputs))
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                             shuffle=True,pin_memory=True)
  optimizerG = torch.optim.Adam(Gnet.parameters(), lr=learning_rate, weight_decay=5e-6)
  optimizerD = torch.optim.Adam(Dnet.parameters(), lr=learning_rate/2, weight_decay=1e-5)
  G_losses = np.zeros(1)
  D_losses = np.zeros(1)
  start = time.time()
  i=0
  for epoch in range(1, epochs+1):
      for batch_idx, (X,y) in enumerate(train_loader):
          Dnet.zero_grad()
          b_size = len(X)
          label = torch.full((b_size,),1,dtype=torch.float,device=device)
          data = X
          # train with real
          # find the predictions f(x) for this batch
          output = Dnet(data.to(device)).view(-1)
          # find the loss
          lossD_real = loss_function(output, label)
          # compute the gradient and update the network parameters
          lossD_real.backward()
          D_x = output.mean().item()
          #train with fake
          noise = torch.randn(b_size,8,device=device)
          fake = Gnet(noise)
          label.fill_(0)
          output = Dnet(fake.detach()).view(-1)
          lossD_fake = loss_function(output, label)
          lossD_fake.backward()
          D_G_z1 = output.mean().item()
          lossD = lossD_real + lossD_fake
          optimizerD.step()
          
          #update G
          Gnet.zero_grad()
          label.fill_(1)
          output = Dnet(fake).view(-1)
          lossG = loss_function(output, label)
          lossG.backward()
          D_G_z2 = output.mean().item()
          optimizerG.step()
      # housekeeping - keep track of our losses and print them as we go
      G_losses = np.append(G_losses,lossG.item())
      D_losses = np.append(D_losses,lossD.item())
      i+=1
      if epoch % print_interval == 0:
          print('[%d/%d][%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epochs, time.time()-start,
                     lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))
  return Gnet,G_losses,Dnet,D_losses

# GAN Fight - run r1 matches of r2 games between d_net (discriminator) and g_net (generator)
def GAN_Fight(d_net,g_net,samples,prms,r1,r2,num_epochs,learn_rate,batch_size):
    crit = nn.BCELoss()
    for i in range(r1):
        noise = torch.randn((samples,prms),device=device)
        noise_dec = g_net(noise)
        noise_lab = torch.cat((noise_dec,torch.zeros((samples,1)).cuda()),1)
        mix_data = torch.cat((noise_lab,real_data_lab),0).cpu().detach().numpy()
        x = torch.FloatTensor(mix_data[:,0:mix_data.shape[1]-1]).cuda().cpu().detach().numpy()
        y = mix_data[:,-1].reshape((mix_data.shape[0],1))
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=7+i)
        
        d_net, losses = train_supervised(x_train, y_train, d_net, num_epochs//4, 
                                         learn_rate, batch_size, crit, p_int)
        plt.plot(losses)
        plt.title('losses: ' + str(i))
        plt.show()
        [MAE_train,MSE_train,RMSE_train,R2_train,
         MAE_test,MSE_test,RMSE_test,R2_test]=model_eval_supervised(x_train,x_test,y_train,y_test,d_net)
        
        for j in range(r2):
            g_net, g_net_losses, d_net, d_net_losses = train_GD(x_real, g_net, 
                                                                d_net, num_epochs, 
                                                                learn_rate, batch_size, 
                                                                crit, p_int)
            plt.plot(g_net_losses,color='green')
            plt.plot(d_net_losses,color='red')
            plt.title('DCGAN losses: ' + str(i) + ', ' + str(j))
            plt.show()
    
    noise = torch.randn((samples,prms),device=device)
    noise_dec = g_net(noise)
    noise_lab = torch.cat((noise_dec,torch.zeros((samples,1)).cuda()),1)
    mix_data = torch.cat((noise_lab,real_data_lab),0).cpu().detach().numpy()
    x = torch.FloatTensor(mix_data[:,0:mix_data.shape[1]-1]).cuda().cpu().detach().numpy()
    y = mix_data[:,-1].reshape((mix_data.shape[0],1))
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5,random_state=17+i)
    
    d_net, losses = train_supervised(x_train, y_train, d_net, num_epochs, 
                                     learn_rate, batch_size, crit, p_int)
    plt.plot(losses)
    plt.title('losses: ' + str(i))
    plt.show()
    [MAE_train,MSE_train,RMSE_train,R2_train,
     MAE_test,MSE_test,RMSE_test,R2_test]=model_eval_supervised(x_train,x_test,
                                                                y_train,y_test,
                                                                d_net)
    return d_net,g_net,MAE_test,MSE_test,RMSE_test,R2_test
