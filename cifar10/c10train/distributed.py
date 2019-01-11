import argparse
from .worker import WorkerTrainer
from .master import MasterTrainer
import os

def main():
  parser = argparse.ArgumentParser(description='Distributed PyTorch CIFAR10 sample worker.')
  parser.add_argument(
    '--gpu',
    type=int,
    default=1,
    help='Use gpu for training')
  parser.add_argument(
    '--batch_size',
    type=int,
    default=1024,
    help='Training mini batch size')

  args = parser.parse_args()

  rank = int(os.environ['RANK'])
  if rank == 0:
    print("Running master...")
    MasterTrainer(args.gpu)
  else:
    print("Running worker...")
    WorkerTrainer(args.gpu, args.batch_size)

if __name__ == '__main__':
  main()
