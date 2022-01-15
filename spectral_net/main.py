import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset
from utils import *
from network import *


def train(model, params, dataset_train):
  to_wandb = params['to_wandb']
  if to_wandb:
    wandb.init(project="SN", entity="bkw1", config=params)

  save_every = params['save_every']
  print_every = params['print_every']
  log_every = params['log_every']
  k = params['k']
  device = params['device'] 
  batch_size = params['batch_size']
  input_sz = params['input_sz']
  
  train_loader_grad = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
  train_loader_ortho = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
  
  optimizer = optim.Adam(model.parameters(), lr=params['lr']) 
  model.A.requires_grad = False
  
  metrics = {"running_loss_sn": 0, "running_loss_ortho": 0, "running_loss": 0, "Accuracy": 0, "epoch": 0}
  for epoch in range(params['n_epochs'] + 1):
  
    metrics['running_loss'] = 0
    metrics['running_loss_sn'] = 0
    metrics['running_loss_ortho'] = 0
    metrics['epoch'] = epoch
      
    model.train()
    for batch_idx, (data_ortho, data_grad) in enumerate(zip(train_loader_ortho, train_loader_grad)):
            # orthogonalization step
            x, target = data_ortho
            x, target = x.to(device), target.to(device)
            x = x.view(x.shape[0], input_sz)
  
            with torch.no_grad():
              res = model(x, ortho_step=True)
            
            # gradient step
            x, target = data_grad
            x, target = x.to(device), target.to(device)
            x = x.view(x.shape[0], input_sz)
  
            # compute similarity matrix for the batch
            with torch.no_grad():
              W = get_affinities(x, params, to_torch=True) 
  
            optimizer.zero_grad()
  
            Y = model(x, ortho_step=False)
            Y_dists = (torch.cdist(Y, Y)) ** 2
            loss_sn = (W * Y_dists).mean() * x.shape[0]
  
            loss = loss_sn
            loss.backward()
  
            with torch.no_grad():
              loss_ortho = (abs((1 / Y.shape[0]) * torch.mm(Y.T, Y) - torch.eye(Y.T.shape[0]))).mean()
              metrics['running_loss'] += loss.item()
              metrics['running_loss_sn'] += loss_sn.item()
              metrics['running_loss_ortho'] += loss_ortho.item()
  
            optimizer.step()
  
    acc = get_cluster_acc(model, dataset_train, k, input_sz)
    if epoch % print_every == 0:
      with torch.no_grad():
        acc = get_cluster_acc(model, dataset_train, k, input_sz)
        metrics['Accuracy'] = acc
        print_metrics(metrics)
        if to_wandb:
          wandb.log(metrics)
  
    elif epoch % log_every == 0:
      with torch.no_grad():
        acc = get_cluster_acc(model, dataset_train, k, input_sz)
        metrics['Accuracy'] = acc
        if to_wandb:
          wandb.log(metrics)
    if acc > params['stop_acc']:
        return
  
  
x_train, x_test, y_train, y_test = generate_cc(n=1200, noise_sigma=0.1, train_set_fraction=1.)
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
dataset_train = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))

device = 'cpu'
#lr = 3 * 1e-8
lr = 3 * 1e-7

params = {
  'k': 2, 
  "n_hidden_1": 1024,
  "n_hidden_2": 512,
  "batch_size": x_train.shape[0],
  "gamma": 23,
  'epsilon': 1e-4,
  "input_sz": x_train.shape[1],
  "affinity": "rbf",
  'lr': lr, 
  'n_epochs' : 5000, 
  'save_every': 50,
  'print_every': 15,
  'log_every': 5,
  'path': "",
  'dataset': "cc",
  "stop_acc": 0.997,
  'to_wandb': False,
  'device': device
}

model = NetOrtho(params)
train(model, params, dataset_train)
plot_clustering(model, x_train, k=params['k'])

