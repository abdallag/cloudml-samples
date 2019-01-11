import torch
from torch import nn


class ConvBNLayer(nn.Module):

  def __init__(self, cin, cout, ksize):
    super(ConvBNLayer, self).__init__()
    self.conv = nn.Conv2d(cin, cout, ksize, padding=1)
    self.bn = nn.BatchNorm2d(cout)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.relu(x)
    return x


class MaxPooledConvBNLayer(nn.Module):

  def __init__(self, cin, cout, ksize):
    super(MaxPooledConvBNLayer, self).__init__()
    self.conv = nn.Conv2d(cin, cout, ksize, padding=1)
    self.bn = nn.BatchNorm2d(cout)
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.relu(x)
    x = self.pool(x)
    return x


class ResidualLayer(nn.Module):

  def __init__(self, cin, cout, ksize):
    super(ResidualLayer, self).__init__()
    self.prep = MaxPooledConvBNLayer(cin, cout, ksize)
    self.res1 = ConvBNLayer(cout, cout, ksize)
    self.res2 = ConvBNLayer(cout, cout, ksize)

  def forward(self, x):
    x = self.prep(x)
    y = self.res1(x)
    y = self.res2(y)
    return x.add(y)


class Classifier(nn.Module):

  def __init__(self):
    super(Classifier, self).__init__()
    self.pool = nn.AdaptiveMaxPool2d(1)
    self.linear = nn.Linear(512, 10)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.pool(x)
    x = x.view(-1, 512)
    x = self.linear(x)
    return x


class ResNet9(nn.Module):

  def __init__(self):
    super(ResNet9, self).__init__()
    self.prep = ConvBNLayer(cin=3, cout=64, ksize=3)
    self.layer1 = ResidualLayer(cin=64, cout=128, ksize=3)
    self.layer2 = MaxPooledConvBNLayer(cin=128, cout=256, ksize=3)
    self.layer3 = ResidualLayer(cin=256, cout=512, ksize=3)
    self.classifier = Classifier()

  def forward(self, x):
    x = self.prep(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.classifier(x)
    return x
