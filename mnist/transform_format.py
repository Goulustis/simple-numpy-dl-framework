import numpy as np
from numpy import genfromtxt
import pandas as pd
import os.path as osp

# dataset csv downloaded from
# https://www.kaggle.com/oddrationale/mnist-in-csv?select=mnist_train.csv

TR_PATH = "csv_format/mnist_train.csv"
TE_PATH = "csv_format/mnist_test.csv"

def process_dataset(dataset, split):
    y = dataset[:,0].reshape(-1, 1)
    imgs = dataset[:,1:]
    imgs = imgs.reshape(-1, 28, 28)

    np.save(osp.join("npy_format", split + "_label.npy"), y)
    np.save(osp.join("npy_format", split + "_test.npy"), imgs)

def main():
    train_data = pd.read_csv(TR_PATH).to_numpy()
    test_data = pd.read_csv(TE_PATH).to_numpy()

    process_dataset(train_data, "train")
    process_dataset(test_data, "test")


if __name__ == "__main__":
    main()

    
