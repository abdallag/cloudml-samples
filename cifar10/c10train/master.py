from collections import defaultdict
from datetime import datetime

import argparse
from queue import Queue
import numpy as np
import os
import torch
from torch import distributed as dist, nn, Tensor
import torchvision
import torchvision.transforms as transforms
from threading import Lock, Thread

from .resnet9 import ResNet9


class MasterTrainer(object):
  def __init__(self, gpu):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                             shuffle=True, num_workers=1)

    self.wd = 0.128
    self.mom = 0.9
    self.lr_schedule = [[0, 5, 24], [0, 0.0016, 0]]
    self.max_epochs = self.lr_schedule[0][-1]

    self.model = ResNet9()
    self.device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
    self.cpu = torch.device("cpu")
    print("Device =%s" % str(self.device))

    if self.device != self.cpu:
      self.model.to(self.device)

    self.param_state = defaultdict(Tensor)

    self.total_training = 0.0
    self.data_size = len(trainset)
    self.all_epochs_data_size = self.data_size * self.max_epochs
    
    self.model_lock = Lock()
    self.params_updated = Queue()
    self.n_reports = 1
    self.done = False

    self.world_size = int(os.environ['WORLD_SIZE'])
    

    self.start_time = datetime.now()
    print("Started at: %s" % str(self.start_time))
    
    dist.init_process_group('gloo')

    t = Thread(target=self.worker_thread)
    t.start()
    while not self.done:
      self.params_updated.get()
      if self.total_training >= self.n_reports * self.data_size:
        self.n_reports += 1
        #Calculate and print test accuracy
        correct = 0
        total = 0
        with torch.no_grad():
          for tdata in testloader:
            timages, tlabels = tdata
            timages = timages.to(self.device)
            tlabels = tlabels.to(self.device)

            self.model_lock.acquire()
            toutputs = self.model(timages)
            self.model_lock.release()

            _, predicted = torch.max(toutputs.data, 1)
            total += tlabels.size(0)
            correct += (predicted == tlabels).sum().item()

          acc = 100 * correct / total
          print('%s: Accuracy of the network after training on %d images= %f %%' %
                  (str(datetime.now() - self.start_time), self.total_training, acc))
    print("Done")


  def update_param(self, name, param, grad, learning_rate):
    #update param
    dp = grad.data
    dp.data.add_(self.wd, param.data)
    b = self.param_state.get(name)
    if b is None:
      b = torch.zeros_like(param.data)
      self.param_state[name] = b
    b.data.mul_(self.mom).add_(dp.data)
    dp.data.add_(self.mom, b.data)
    param.data.add_(-learning_rate, dp.data)


  def worker_thread(self):
    print("Starting worker thread")
    while not self.done:
      with torch.no_grad():
        for i in range(1, self.world_size):      
          # Receive batch size
          dist_batch = torch.zeros(1)
          dist.recv(dist_batch, i)

          params_to_send = []
            
          batch_size = dist_batch.item()
          if batch_size != 0:
            #H ow far are we?
            self.total_training += batch_size
            if self.total_training >= self.all_epochs_data_size - 1:
              self.done = True

            # compute adaptive learning rate
            x = self.total_training/self.data_size
            learning_rate = np.interp(x, self.lr_schedule[0], self.lr_schedule[1])
            print("%s: Grads from w%d: images_trained=%f lr=%f"
                  % (str(datetime.now() - self.start_time), i, self.total_training, learning_rate))

            self.model_lock.acquire()
            for name, param in self.model.named_parameters():
              # Receive grad an update param
              grad = torch.zeros_like(param.to(self.cpu))
              dist.recv(grad, i)
              grad = grad.to(self.device)
                  
              
              self.update_param(name, param, grad, learning_rate)
              params_to_send.append(param.to(self.cpu))

            self.model_lock.release()

            self.params_updated.put(0)
          else:
            self.model_lock.acquire()
            for param in self.model.parameters():
              params_to_send.append(param.to(self.cpu))

            self.model_lock.release()

          # Send updated param
        for i in range(1, self.world_size):
          for param in params_to_send:
            dist.send(param, i)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Distributed PyTorch CIFAR10 sample worker.')
  parser.add_argument(
    '--gpu',
    type=int,
    default=0,
    help='Use gpu for training')

  args = parser.parse_args()
  MasterTrainer(args.gpu)
