
__all__ = ['halfband1d','residual1d','upsample1d','downsample1d']

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, filtfilt
from scipy.signal.windows import blackman, hann
import tensorflow as tf
import tensorflow.keras as tfk

# Windowed sinc kernel
kernel_size = 15
t = np.arange(-(kernel_size//2),(kernel_size+1)//2) / 2
kernel = np.sin(t*np.pi)/(t*np.pi)
off = kernel_size//2
kernel[off] = 1.0
kernel *= hann(kernel_size)
kernel = tf.constant(kernel.reshape((-1,1,1)), dtype=tf.float32)

@tf.function
def halfband1d(x,scale=1):
    # if tf.shape(tf.shape(x))[0]==1:
    #     x = tf.reshape(x, (1,-1,1))
    # if tf.shape(tf.shape(x))[0]==2:
    #     x = tf.expand_dims(x, -1)
    y = tf.nn.conv1d(x, filters=kernel*scale,
                      stride=1, padding='SAME')
    return y

@tf.function
def residual1d(x,scale=1):
    # if tf.shape(tf.shape(x))[0]==1:
    #     x = tf.reshape(x, (1,-1,1))
    # if tf.shape(tf.shape(x))[0]==2:
    #     x = tf.expand_dims(x, -1)
    return x - halfband1d(x,0.5)

@tf.function
def upsample1d(x):
    # if tf.shape(tf.shape(x))[0]==1:
    #     x = tf.reshape(x, (1,-1,1))
    # if tf.shape(tf.shape(x))[0]==2:
    #     x = tf.expand_dims(x, -1)
    s = tf.shape(x)+tf.shape(x)*[0,1,0]
    x = tf.expand_dims(x,-1)
    x = tf.concat([x, tf.zeros_like(x)],axis=-2)
    return halfband1d(tf.reshape(x, s))

@tf.function
def downsample1d(x):
    # if tf.shape(tf.shape(x))[0]==1:
    #     x = tf.reshape(x, (1,-1,1))
    # if tf.shape(tf.shape(x))[0]==2:
    #     x = tf.expand_dims(x, -1)
    y = halfband1d(x,0.5)
    return y[:,::2,:]

def testing1(x):
    print(x.shape)
    y = halfband1d(x,0.5)
    print(y.shape)
    plt.clf()
    plt.plot(x)
    plt.plot(lfilter(np.squeeze(kernel)/2,1,np.hstack([x,np.zeros(off)]))[off:])
    plt.plot(tf.reshape(y,(-1,)), alpha=0.5)

def testing2(x):
    print(x.shape)
    y = upsample1d(x)
    print(y.shape)
    plt.clf()
    plt.plot(np.squeeze(y))

def testing3(x):
    print(x.shape)
    y = downsample1d(x)
    print(y.shape)
    plt.clf()
    plt.plot(np.squeeze(y))

def testing4(x):
    y0 = residual1d(x)
    y1 = upsample1d(downsample1d(x))
    plt.clf()
    plt.plot(np.squeeze(x))
    plt.plot(np.squeeze(y0))
    plt.plot(np.squeeze(y1))

def testing():
    N = 4096
    M = N//2
    x0 = np.arange(0,N,2)
    #y0 = np.random.normal(0,1,N)
    y0 = np.sin(np.linspace(0,1,N)*np.pi*np.logspace(np.log10(20),np.log10(22000),N)+np.pi/8)
    # y0 = np.sin(np.linspace(0,1,N)*np.pi*10+np.pi/8)
    y0 = y0.astype(np.float32)
    testing4(y0.reshape((1,-1,1)))

def testing():
    B = kernel
    N = 4096
    M = N//2
    x0 = np.arange(0,N,2)
    #y0 = np.random.normal(0,1,M)
    y0 = np.sin(np.linspace(0,1,M)*np.pi*np.logspace(np.log10(20),np.log10(22000),M)+np.pi/8)

    # ZoH
    x1 = np.arange(N)
    y1 = np.hstack([y0.reshape((-1,1))]*2).reshape((-1,))

    # Windowed sinc upsampling
    B = np.sin(t*np.pi)/(t*np.pi)
    off = kernel_size//2
    B[off] = 1.0
    B *= hann(kernel_size)

    x2 = np.arange(N)
    y2 = np.hstack([y0,np.zeros(off//2)+y0[-1]])
    y2 = np.hstack([y2.reshape((-1,1)),np.zeros((M+off//2,1))]).reshape((-1,))
    y2 = lfilter(B, [1], x=y2)[off:]

    # Windowed sinc downsampling
    x3 = np.arange(0,N,2)
    y3 = np.hstack([y2,np.zeros(off)+y2[-1]])
    y3 += np.sin(np.arange(N+off)/44100*np.pi*2*15000)*0.3
    y4 = y3[:-off:2]
    y3 = lfilter(B/2, [1], x=y3)[off::2]

    plt.clf()
    plt.subplot(311)
    # plt.plot(x0,y0,'-',label='orig')
    plt.plot(x1,y1,label='ZoH')
    plt.plot(x2,y2,'-',label='upsampled')
    # plt.plot(x3,y4,'-',label='unfiltered')
    # plt.plot(x3,y3,'-',label='filtered')
    plt.legend()

    plt.subplot(312)
    F = lambda x: 20*np.log10(np.abs(np.fft.rfft(x*hann(x.shape[0]))))
    ff = lambda t: np.linspace(0,np.pi,kernel_size//2+1)
    plt.plot(ff(y0),F(y0),'-',label='orig')
    plt.plot(ff(y1),F(y1),label='ZoH')
    plt.plot(ff(y2),F(y2),'-',label='upsampled')
    plt.legend()

    plt.subplot(313)
    plt.plot(ff(y4),F(y4),'-',label='unfiltered')
    plt.plot(ff(y3),F(y3),'-',label='filtered')
    plt.legend()
    plt.show()


    # Equivalent filtering in tensorflow
    y5 = tf.constant(y0.reshape((1,-1,1)),tf.float32)
    y5 = tf.concat([tf.zeros((1,off,1),dtype=tf.float32), y5],axis=1)
    print(y5.shape)
    y5 = tf.nn.conv1d(y5, filters=tf.constant(B.reshape((-1,1,1))/2,tf.float32),
                      stride=1, padding='VALID')
    print(y5.shape)
    plt.clf()
    plt.plot(y0)
    plt.plot(lfilter(B/2,1,np.hstack([y0,np.zeros(off)]))[off:])
    plt.plot(tf.reshape(y5,(-1,)), alpha=0.5)
