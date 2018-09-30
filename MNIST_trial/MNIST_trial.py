import numpy as np
import os
import gzip
import struct
import logging
import mxnet as mx
import matplotlib.pyplot as plt

logging.getLogger().setLevel(logging.DEBUG)

TrainLabelFileName = 'train-labels-idx1-ubyte.gz'
TrainImageFileName = 'train-images-idx3-ubyte.gz'
TestLabelFileName = 't10k-labels-idx1-ubyte.gz'
TestImageFileName = 't10k-images-idx3-ubyte.gz'
AddedPath = '../Dataset/'

# Get Training data from input data set
# label_url = input label file path
# image_url = input image file path
def ReadData(label_url, image_url):
    with gzip.open(label_url) as fLabel:
        magic, num = struct.unpack(">II", fLabel.read(8))
        label = np.fromstring(fLabel.read(), dtype=np.int8)

    with gzip.open(image_url, 'rb') as fImg:
        magic, num, rows, cols = struct.unpack(">IIII", fImg.read(16))
        image = np.fromstring(fImg.read(), dtype=np.uint8)
        image = image.reshape(len(label), 1, rows, cols)
        image = image.astype(np.float32)/255.0

    return (label, image)

# Read training data
Train_lbl, Train_img = ReadData(AddedPath + TrainLabelFileName, AddedPath + TrainImageFileName)
Validation_lbl, Validation_img = ReadData(AddedPath + TestLabelFileName, AddedPath + TestImageFileName)

BatchSize = 32

# define Training iteritor
train_iter = mx.io.NDArrayIter(Train_img, Train_lbl, BatchSize, True)
val_iter = mx.io.NDArrayIter(Validation_img, Validation_lbl, BatchSize)

# test: show up images
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(Train_img[i].reshape(28, 28), cmap='Greys_r')
    plt.axis('off')

print('label: %s' % (Train_lbl[0:10],))
plt.show()

print('\nThe trainer begins to work')

# define Network
data = mx.symbol.Variable('data')

# Turn image into 1-D array data(Flatten)
flatten = mx.sym.Flatten(data=data, Name="flatten")

# 128 neurons fully connected, ReLU
fc1 = mx.sym.FullyConnected(data = flatten, num_hidden=128, Name="fc1")
act1 = mx.sym.Activation(fc1, "relu", "act1")

# 64 neurons fully connected, ReLU
fc2 = mx.sym.FullyConnected(act1, num_hidden=64, name = "fc2")
act2 = mx.sym.Activation(fc2, "relu", "act2")

# Output layer, fully connected, 10 neurons since 10 categories to classify
fc3 = mx.sym.FullyConnected(act2, num_hidden=10, name="fc3")
# Do SoftMax to get 
net = mx.sym.SoftmaxOutput(fc3, name = "softmax")

# Using MX module to show network structure
shape = {"data": (BatchSize, 1, 28, 28)}
mx.visualization.print_summary(net, shape)

mx.visualization.plot_network(symbol=net, shape=shape).view()



