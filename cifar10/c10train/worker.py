from collections import defaultdict
from datetime import datetime

import argparse
import numpy as np
import os
import torch
from torch import distributed as dist, nn, Tensor
import torchvision
import torchvision.transforms as transforms

from .resnet9 import ResNet9

class WorkerTrainer(object):
  def __init__(self, gpu, batch_size):

    world_size = int(os.environ['WORLD_SIZE']) -1 #execluding master
    rank = int(os.environ['RANK']) -1
    self.gpu = gpu

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    loss_fn = nn.CrossEntropyLoss()
    self.model = ResNet9()
    start_time = datetime.now()
    print("Started at: %s" % str(start_time))

    dist.init_process_group('gloo')
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    
    dist_sampler = torch.utils.data.DistributedSampler(trainset, num_replicas = world_size, rank=rank)
    batch_sampler = torch.utils.data.BatchSampler(dist_sampler, batch_size=batch_size, drop_last=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_sampler=batch_sampler, num_workers=4)
    
    self.device = torch.device("cuda:0" if self.gpu and torch.cuda.is_available() else "cpu")
    self.cpu = torch.device("cpu")
    print("Device =%s" % str(self.device))

    if self.device != self.cpu:
      self.model.to(self.device)
    else:
      self.gpu = False

    max_epochs = 24

    update_every = 5
    t = 0
    total_training = 0.0

    self.dist_batch = 0.0
    self.update_params()

    for ep in range(max_epochs):
      loss_sum = 0.0
      dist_sampler.set_epoch(ep)

      for i, data in enumerate(trainloader, 0):
        t += 1
        total_training += batch_size
        self.dist_batch += batch_size

        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)
        y_pred = self.model(x)

        loss = loss_fn(y_pred, y)

        # print progress
        loss_sum += loss.item()
        if i % update_every == update_every -1:
          print("%s: trained %5d t=%5d loss= %.3f" % (str(datetime.now() - start_time), total_training, i+1, loss_sum / update_every))
          loss_sum = 0.0

        # calculate grads
        loss.backward()

        self.update_params()

      print("%d: trained using %d images in %d iterations" % (ep, total_training, t))

    print("Done.")


  def update_params(self):
    #print('Updating params...')
    with torch.no_grad():
      # Send batch size
      dist.send(torch.tensor(self.dist_batch), 0)

      # Send grads if we have any
      if self.dist_batch != 0:
        for param in self.model.parameters():
          grad = param.grad.to(self.cpu)
          dist.send(grad.data, 0)  

      for param in self.model.parameters():
        if self.gpu:
          rparam = torch.zeros_like(param.to(self.cpu))  
          dist.recv(rparam, 0)
          param.data = rparam.to(self.device)
        else:
          dist.recv(param.data, 0)

    self.model.zero_grad()
    self.dist_batch = 0.0
    #print('Done updating')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Distributed PyTorch CIFAR10 sample worker.')
  parser.add_argument(
    '--gpu',
    type=int,
    default=0,
    help='Use gpu for training')
  parser.add_argument(
    '--batch_size',
    type=int,
    default=256,
    help='Training mini batch size')

  args = parser.parse_args()
  print(args.gpu)
  WorkerTrainer(args.gpu, args.batch_size)
