#!/usr/bin/env python3

import sounddevice as sd
import numpy as np
import sys, os, time
import matplotlib.pyplot as plt
import wave, struct

fn = 'x_output.npy'
if len(sys.argv) > 1:
    fn = sys.argv[1]

x = np.load(fn)

def wrwav(i):
    with wave.open(f'x_output_{i}.wav','wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.setnframes(x.shape[1])
        w.writeframesraw((np.clip(x[i],-1,1)*32767).astype(np.int16).tostring())

plt.ion()
plt.clf()
p1 = plt.plot(x[0], 'k', alpha=0.2)[0]
p2 = plt.plot(np.clip(x[0],-1,1), 'k')[0]
plt.ylim(x.min(), x.max())
for i in range(x.shape[0]):
    p1.set_ydata(x[i])
    p2.set_ydata(np.clip(x[i],-1,1))
    #sd.play(np.clip(x[i],-1,1), 44100)
    wrwav(i)
    sd.play(x[i]/max(x[i].max(),-x[i].min()), 44100)
    plt.pause(0.5)
