
# A dataset of kick drum sounds.
#
# Stephen Sinclair <radarsat1@gmail.com>

import matplotlib as mpl
mpl.use('Agg')
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif'] = ['cmr10']
# mpl.font_manager.findfont('cmmr')
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys, os
import pkg_resources

fn = pkg_resources.resource_filename(__name__, 'kicks_dataset.hdf5')

class Dataset(object):
    def __init__(self, params):
        self.params = params
        L = None
        if 'data_dim' in params:
            L = params['data_dim']
        with h5py.File(fn, 'r') as h:
            self.kicks = h['data'][:,:L].astype(np.float32)
        self.kicks *= 1.0/np.maximum(self.kicks.max(), -self.kicks.min())
        self.length = self.kicks.shape[1]

    def latent_generator(self, batch_size):
        if self.params['latent_prior']=='uniform':
            while True:
                yield tf.random.uniform([batch_size, self.params['latent_dim']])
        elif self.params['latent_prior']=='normal':
            while True:
                yield tf.random.normal([batch_size, self.params['latent_dim']])
        else:
            assert False and 'Unknown latent prior.'

    def data_generator(self, batch_size):
        while True:
            i = np.random.randint(0, self.kicks.shape[0], (batch_size,))
            yield self.kicks[i]

    def eps_generator(self, batch_size, dims):
        while True:
            yield tf.random.normal([batch_size, dims])

    # Visualization
    def init_viz(self, dirname, method, variant, x_input, z_input, eps_input):
        self.dirname = dirname
        self.method = method
        self.variant = variant
        self.fig, (self.ax1,self.ax2,self.ax3) = plt.subplots(1,3, num=1, figsize=(9,4))
        self.lims = None
        self.frame = 0
        self.x_input_viz = x_input
        self.eps_input_viz = eps_input
        self.z_input_viz = z_input
        self.plt_sample = self.ax1.plot(x_input[0]*0)[0]
        self.scat_z = self.ax2.scatter(z_input[:,0], z_input[:,1])
        self.plt_orig = self.ax3.plot(x_input[0])[0]
        self.plt_recon = self.ax3.plot(x_input[0]*0)[0]
        self.ax1.set_title('decoded', fontname='cmr10')
        self.ax3.set_title('reconstruction', fontname='cmr10')
        if self.params['latent_prior']=='normal':
            self.ax2.plot(np.cos(np.linspace(0,2*np.pi,200))*1.96,
                          np.sin(np.linspace(0,2*np.pi,200))*1.96, 'k--', alpha=0.4)
            self.ax2.set_xlim(-3,3)
            self.ax2.set_ylim(-3,3)
        elif self.params['latent_prior']=='uniform':
            self.ax2.plot([[0,1],[0,1],[0,0],[1,1]],
                          [[0,0],[1,1],[0,1],[0,1]], 'k--', alpha=0.4)
            self.ax2.set_xlim(-0.5,1.5)
            self.ax2.set_ylim(-0.5,1.5)
        if self.z_input_viz.shape[1]==2:
            self.ax2.set_title('encoded', fontname='cmr10')
        else:
            self.ax2.set_title('encoded (first 2 dims)', fontname='cmr10')

        if not os.path.exists(self.dirname):
            os.mkdir(self.dirname)
        plt.subplots_adjust()

    def viz(self, epoch, decoder, encoder):
        continue_seed = int(tf.random.uniform([1])*65536)
        tf.random.set_seed(2)
        z_output = encoder(self.x_input_viz)
        x_output = decoder([self.z_input_viz,self.eps_input_viz])
        yl_ = np.array([np.minimum(np.min(self.x_input_viz),np.min(x_output)),
                        np.maximum(np.max(self.x_input_viz),np.max(x_output))])
        if self.lims is None:
            self.lims = yl_
        else:
            self.lims = self.lims*0.95+yl_*0.05
        self.ax1.set_ylim(self.lims)
        self.ax3.set_ylim(self.lims)

        self.plt_sample.set_ydata(x_output[0])
        self.scat_z.set_offsets(z_output)

        x_output = decoder([z_output[:1],self.eps_input_viz[:1]])
        self.plt_recon.set_ydata(x_output[0])

        self.fig.suptitle(
            f'{self.z_input_viz.shape[1]}D {self.method} {self.variant}: Epoch {epoch}',
            fontname='cmr10')
        self.fig.savefig('now.png')
        self.fig.savefig(f'{self.dirname}/frame{self.frame:06d}.png')
        self.frame += 1
        tf.random.set_seed(continue_seed)

if __name__=='__main__':
    d = Dataset({'latent_prior':'normal', 'data_dim': 2, 'latent_dim': 2})
    x = next(d.data_generator(16))
    z = next(d.latent_generator(16))
    e = next(d.eps_generator(16,1))
    d.init_viz('testviz', 'testmethod', 'testvariant', x, z, e)
    d.viz(0, lambda z: x, lambda x: z)