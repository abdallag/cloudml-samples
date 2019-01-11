from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

from .resnet9 import ResNet9


def main():
  batch_size = 256

  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  trainset = torchvision.datasets.CIFAR10(
      root='./data', train=True, download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(
      trainset, batch_size=batch_size, shuffle=True, num_workers=8)

  testset = torchvision.datasets.CIFAR10(
      root='./data', train=False, download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(
      testset, batch_size=batch_size, shuffle=True, num_workers=8)

  loss_fn = nn.CrossEntropyLoss()
  wd = 5e-4 * batch_size
  mom = 0.9
  lr_schedule = [[0, 5, 25], [0, 0.0016, 0]]

  model = ResNet9()

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)
  model.to(device)

  start_time = datetime.now()
  print('Started at: %s' % str(start_time))

  update_every = 20
  param_state = defaultdict(torch.Tensor)
  max_epochs = lr_schedule[0][-1]
  data_size = len(trainset)
  total_training = 0.0
  t = 0

  for ep in range(max_epochs):
    loss_sum = 0.0

    for i, data in enumerate(trainloader, 0):
      t += 1
      total_training += batch_size
      x = total_training / data_size
      lr = np.interp(x, lr_schedule[0], lr_schedule[1])

      x, y = data
      x = x.to(device)
      y = y.to(device)

      y_pred = model(x)

      loss = loss_fn(y_pred, y)

      loss_sum += loss.item()
      if i % update_every == update_every - 1:
        print('%s: trained %5d t=%5d loss= %.3f using lr=%f' %
              (str(datetime.now() - start_time), total_training, i + 1,
               loss.item(), lr))

        loss_sum = 0.0

      model.zero_grad()
      loss.backward()

      with torch.no_grad():
        for name, param in model.named_parameters():
          # Nestrov
          dp = param.grad.data
          dp.data.add_(wd, param.data)
          b = param_state.get(name)
          if b is None:
            b = torch.zeros_like(param.data)
            param_state[name] = b
          b.data.mul_(mom).add_(dp.data)
          dp.data.add_(mom, b.data)
          param.data.add_(-lr, dp.data)

    print('%d: trained using %d images in %d iterations' % (ep, total_training,
                                                            t))

    # Calculate and plot test accuracy
    correct = 0
    total = 0
    with torch.no_grad():
      for tdata in testloader:
        timages, tlabels = tdata
        timages = timages.to(device)
        tlabels = tlabels.to(device)
        toutputs = model(timages)
        _, predicted = torch.max(toutputs.data, 1)
        total += tlabels.size(0)
        correct += (predicted == tlabels).sum().item()

    acc = 100 * correct / total
    print('%s: Accuracy of the network on the %d test images= %f %%' %
          (str(datetime.now() - start_time), total, acc))


if __name__ == '__main__':
  main()
