#!/usr/bin/env python3

import numpy as np
import h5py
import sys, os, time
import matplotlib.pyplot as plt
import wave, struct

fn = 'x_output.npy'
if len(sys.argv) > 1:
    fn = sys.argv[1]

x = np.load(fn)

with h5py.File('datasets/kicks_dataset.hdf5') as h:
    pcm = np.copy(h['data'][:,:x.shape[1]])
    pcm = pcm/np.maximum(pcm.max(),-pcm.min())

F = 20*np.log10(np.abs(np.fft.rfft(pcm,axis=1))+1e-10)
X = 20*np.log10(np.abs(np.fft.rfft(x,axis=1))+1e-10)
closest = np.zeros(X.shape[0],dtype=int)
similarity_cos = lambda F,X: np.dot(F,X)/(np.linalg.norm(F,axis=1)*np.linalg.norm(X))
similarityL2 = lambda F,X: -np.mean((F-X)**2,axis=1)
for i in range(X.shape[0]):
    closest[i] = np.argmax(similarityL2(F,X[i]))

plt.clf()
for j in range(x.shape[0]):
    plt.subplot(4,4,j*2+1)
    plt.plot(x[j])
    plt.plot(pcm[closest[j]])
    plt.subplot(4,4,j*2+2)
    plt.plot(X[j])
    plt.plot(F[closest[j]])
plt.show()
