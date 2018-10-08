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
#AddedPath = '../F_mnist/Dataset/'


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

# Generate a FullyConnected network layer
def GenerateFCLayers(data, num_hidden, act_Func, layer_name):
    fc = mx.sym.FullyConnected(data=data, num_hidden=num_hidden, name=layer_name)
    act = mx.sym.Activation(data=fc, act_type=act_Func, name=layer_name + '_act')
    return act

# Read training data
Train_lbl, Train_img = ReadData(AddedPath + TrainLabelFileName, AddedPath + TrainImageFileName)
Validation_lbl, Validation_img = ReadData(AddedPath + TestLabelFileName, AddedPath + TestImageFileName)

BatchSize = 32

# define Training iteritor
train_iter = mx.io.NDArrayIter(Train_img, Train_lbl, BatchSize, shuffle=False)
val_iter = mx.io.NDArrayIter(Validation_img, Validation_lbl, BatchSize)

# test: show up images
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(Train_img[i].reshape(28, 28), cmap='Greys_r')
    plt.axis('off')

print('label: %s' % (Train_lbl[0:10],))
plt.show()

print('\nThe trainer begins to work')


#try:
# define Network
data = mx.symbol.Variable('data')
   
# Turn image into 1-D array data(Flatten)
net = mx.sym.Flatten(data=data, Name="flatten")
    
# 96 neurons fully connected, ReLU
net = GenerateFCLayers(net, 96, 'relu', 'FC_1')
    
# 96 neurons fully connected, ReLU
net = GenerateFCLayers(net, 96, 'relu', 'FC_2')

# 32 neurons fully connected, ReLU
net = GenerateFCLayers(net, 32, 'relu', 'FC_3')
    
# Output layer, fully connected, 10 neurons since 10 categories to classify
net = mx.sym.FullyConnected(net, num_hidden=10, name="fc_out")
# Do SoftMax to get 
net = mx.sym.SoftmaxOutput(net, name = "softmax")
    
# Using MX module to show network structure
shape = {"data": (BatchSize, 1, 28, 28)}
mx.visualization.print_summary(net, shape)
    
mx.visualization.plot_network(symbol=net, shape=shape).view()

# start training
module = mx.mod.Module(symbol=net)
module.fit(train_iter, 
           eval_data=val_iter,
           optimizer='sgd',
           optimizer_params= {'learning_rate':0.2, 'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=60000/BatchSize, factor=0.9)},
           num_epoch=20,
           batch_end_callback=mx.callback.Speedometer(BatchSize, 60000/BatchSize)
)

module.save_params('FC_mnist_test1')


#except Exception:
#    print(Exception.next())



