
# Steve's adaptation of VEEGAN example:
# https://github.com/akashgit/VEEGAN/blob/master/VEEGAN_2D_RING.ipynb
# to TensorFlow 2 + Keras, with the TensorFlow Probabilities library:
# https://blog.tensorflow.org/2019/03/variational-autoencoders-with.html
#
# Stephen Sinclair <radarsat1@gmail.com>

import matplotlib as mpl
mpl.use('Agg')
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif'] = ['cmr10']
# mpl.font_manager.findfont('cmmr')
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from tqdm import tqdm
import sys, os
Dense = tfk.layers.Dense

dists = tfp.distributions

# Generate 2D grid

mus = np.array([np.array([i, j]) for i, j in product(range(-4, 5, 2),
                                                     range(-4, 5, 2))],
               dtype=np.float32)
# mus = np.array([np.array([i, j]) for i, j in product(range(-1, 1, 1),
#                                                      range(-1, 1, 1))],
#                dtype=np.float32)

class Dataset(object):
    def __init__(self, params):
        self.params = params

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
            yield self.normal_mixture([batch_size, self.params['data_dim']])

    def eps_generator(self, batch_size, dims):
        while True:
            yield tf.random.normal([batch_size, dims])

    def normal_mixture(self, shape, **kwargs):
        return self.create_distribution(shape[0],25,shape[1],**kwargs)

    @tf.function
    def create_distribution(self, batch_size, num_components=25,
                            num_features=2,**kwargs):
        num_components = len(mus)
        cat = dists.Categorical(tf.zeros(num_components, dtype=np.float32))

        s = 0.05
        sigmas = [np.array([s,s]).astype(np.float32) for i in range(num_components)]
        components = list((dists.MultivariateNormalDiag(mu, sigma) 
                           for (mu, sigma) in zip(mus, sigmas)))
        data = dists.Mixture(cat, components)
        return data.sample(batch_size)

    # Network definitions
    def normal_mixture(self, shape, **kwargs):
        return self.create_distribution(shape[0],25,shape[1],**kwargs)

    # Visualization
    def init_viz(self, dirname, method, variant, x_input, z_input, eps_input):
        self.dirname = dirname
        self.method = method
        self.variant = variant
        self.fig, (self.ax,self.ax2,self.ax3) = plt.subplots(1,3, num=1, figsize=(9,4))
        self.lims = None
        self.frame = 0
        self.x_input_viz = x_input
        self.eps_input_viz = eps_input
        self.z_input_viz = z_input
        # TODO abstract the possibility of a pre-clustered input distribution
        N = tf.shape(self.z_input_viz)[0]
        self.x_input_viz = np.array(
            [np.hstack([tf.random.normal([N,1], mean=m[0], stddev=0.05),
                        tf.random.normal([N,1], mean=m[1], stddev=0.05)])
             for m in mus])
        self.eps_input_viz = next(self.eps_generator(500, 1))
        self.z_input_viz = next(self.latent_generator(500))

        self.x_scat = self.ax.scatter([0],[0], label='target', marker='.', alpha=0.05, edgecolors='none')
        self.px_scat = self.ax.scatter([0],[0], label='generated', marker='.', alpha=0.2, edgecolors='none')
        if not os.path.exists(self.dirname):
            os.mkdir(self.dirname)
        plt.subplots_adjust()

    def viz(self, epoch, decoder, encoder):
        continue_seed = int(tf.random.uniform([1])*65536)
        tf.random.set_seed(2)
        N = self.x_input_viz.shape[1]
        D = self.x_input_viz.shape[2]
        x_output = decoder([self.z_input_viz,self.eps_input_viz])
        xl_ = np.array([np.minimum(np.min(self.x_input_viz[:,:,0]),np.min(x_output[:,0])),
                        np.maximum(np.max(self.x_input_viz[:,:,0]),np.max(x_output[:,0]))])
        yl_ = np.array([np.minimum(np.min(self.x_input_viz[:,:,1]),np.min(x_output[:,1])),
                        np.maximum(np.max(self.x_input_viz[:,:,1]),np.max(x_output[:,1]))])
        if self.lims is None:
            self.lims = np.hstack([xl_, yl_])
        else:
            self.lims[0] = self.lims[0]*0.95+xl_[0]*0.05
            self.lims[1] = self.lims[1]*0.95+xl_[1]*0.05
            self.lims[2] = self.lims[2]*0.95+yl_[0]*0.05
            self.lims[3] = self.lims[3]*0.95+yl_[1]*0.05
        self.ax.set_xlim(self.lims[:2]); self.ax.set_ylim(self.lims[2:4])
        self.x_scat.set_offsets(tf.reshape(self.x_input_viz,(len(mus)*N,D))[:,:2])
        self.px_scat.set_offsets(x_output[:,:2])
        f = self.ax.set_title('decoded', fontname='cmr10')
        #ax.legend(loc=1, prop=f.get_fontproperties())
        self.ax2.clear()
        self.ax3.clear()
        # ax2.scatter(self.z_input_viz[:,0], self.z_input_viz[:,1])
        for i,m in enumerate(mus):
            z_output = encoder(self.x_input_viz[i])
            self.ax2.scatter(z_output[:,0], z_output[:,1], marker='.', alpha=0.2, edgecolors='none')
            x_output = decoder([z_output,self.eps_input_viz[i:i+N]])
            self.ax3.scatter(x_output[:,0], x_output[:,1], marker='.', alpha=0.2, edgecolors='none')
        self.ax3.set_xlim(self.lims[:2]); self.ax3.set_ylim(self.lims[2:4])
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
        self.fig.suptitle(
            f'{self.z_input_viz.shape[1]}D {self.method} {self.variant} — Epoch {epoch}',
            fontname='cmr10')
        # fig.canvas.draw()
        # plt.pause(0.0001)
        self.fig.savefig('now.png')
        self.fig.savefig(f'{self.dirname}/frame{self.frame:06d}.png')
        self.frame += 1
        tf.random.set_seed(continue_seed)

if __name__=='__main__':
    d = Dataset({'latent_prior':'normal', 'data_dim': 2, 'latent_dim': 2})
    x = next(d.data_generator(200))
    z = next(d.latent_generator(200))
    e = next(d.eps_generator(200,1))
    d.init_viz('testviz', 'testmethod', 'testvariant', x, z, e)
    d.viz(0, lambda z: x, lambda x: z)
