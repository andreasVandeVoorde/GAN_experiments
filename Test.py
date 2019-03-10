import os, time, itertools, imageio, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])
mnist2 = np.array([x for (x,y) in zip(mnist.train.images,mnist.train.labels) if y[7]!=0])
some = tf.convert_to_tensor(mnist2, np.float32)
print(mnist.train.images.shape[0])
print(some.shape[0])

