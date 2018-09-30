import numpy as np
import os
import gzip
import struct
import logging
import mxnet as mx
import matplotlib.pyplot as plt

logging.getLogger().setLevel(logging.DEBUG)

TrainLabelPath = 'train-labels-idx1-ubyte.gz'
TrainImagePath = 'train-images-idx3-ubyte.gz'
TestLabelPath = 't10k-labels-idx1-ubyte.gz'
TestImagePath = 't10k-images-idx3-ubyte.gz'

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
Train_lbl, Train_img = ReadData(TrainLabelPath, TTrainImagePath)
Validation_lbl, Validation_img = ReadData(TestLabelPath, TestImagePath)

BatchSize = 32

# define Training iteritor
train_iter = mx.io.NDArrayIter(Train_img, Train_lbl, BatchSize, True)
val_iter = mx.io.NDArrayIter(Validation_img, Validation_lbl, BatchSize)

# test: show up images
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(Train_img[i].reshape(28, 28), cmap='Greys_r')
    plt.axis('off')

plt.show()
print('label: %s' % (Train_lbl[0:10],))

