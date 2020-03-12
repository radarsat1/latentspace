
# Steve's adaptation of VEEGAN example:
# https://github.com/akashgit/VEEGAN/blob/master/VEEGAN_2D_RING.ipynb
# to TensorFlow 2 + Keras, with the TensorFlow Probabilities library:
# https://blog.tensorflow.org/2019/03/variational-autoencoders-with.html
#
# Stephen Sinclair <radarsat1@gmail.com>

# Google Colab version here:
# https://colab.research.google.com/drive/1dSW7Yn8okef-ng-Inkev205L2Gy6_MIx

# 12/03/2020: Added switches for BiGAN and Vanilla GAN for comparison

# Imports

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

ds = tfp.distributions

use_veegan = True
use_bigan = False
use_vanillagan = False

# Parameters

# Note: potentially confusing; what VEEGAN calls "latent" is actually
# the data space, and the "latent" space is "input_dim".  I kept this
# from the VEEGAN example code but it's a bit weird, I believe it's
# because they consider the structure of the GAN as an autoencoder
# from input x to p(x|z), where the latent space of that autoencoder,
# q(z|x), would be the data; this is the reverse of a BiGAN which
# would decode data p(x|z) from input z and infer q(z|x) from p(x).
params = {
    'batch_size': 500,
    'latent_dim': 2, # actually data dims!
    'eps_dim': 1, 
    'input_dim': 50, # actually latent dims!
    'n_layer_disc': 2,
    'n_hidden_disc': 128,
    'n_layer_gen': 2,
    'n_hidden_gen': 128,
    'n_layer_inf': 2,
    'n_hidden_inf': 128,
    'learning_rate': 1e-3,
}

if use_veegan: method = 'VEEGAN'
elif use_bigan: method = 'BiGAN'
elif use_vanillagan: method = 'Vanilla GAN'

# Generate 2D grid

@tf.function
def create_distribution(batch_size, num_components=25, num_features=2,**kwargs):
    cat = ds.Categorical(tf.zeros(num_components, dtype=np.float32))
    mus = np.array([np.array([i, j]) for i, j in product(range(-4, 5, 2),
                                                         range(-4, 5, 2))],
                   dtype=np.float32)

    s = 0.05
    sigmas = [np.array([s,s]).astype(np.float32) for i in range(num_components)]
    components = list((ds.MultivariateNormalDiag(mu, sigma) 
                       for (mu, sigma) in zip(mus, sigmas)))
    data = ds.Mixture(cat, components)
    return data.sample(batch_size)

# Network definitions

def normal_mixture(shape, **kwargs):
    return create_distribution(shape[0],25,shape[1],**kwargs)

def generative_network(batch_size, latent_dim, input_dim, n_layer,
                       n_hidden, eps=1e-6, X=None):
    z = tfk.layers.Input((latent_dim,), name="p_z")
    h = tfk.layers.Dense(n_hidden, activation='relu')(z)
    h = tfk.layers.Dense(n_hidden, activation='relu')(h)
    p = tfk.layers.Dense(input_dim)(h)
    h = tfk.layers.Dense(n_hidden, activation='relu')(z)
    h = tfk.layers.Dense(n_hidden, activation='relu')(h)
    s = tfk.layers.Dense(input_dim, activation='softplus')(h)
    x = tfp.layers.DistributionLambda(
        lambda x: tfp.distributions.Normal(loc=x[0], scale=x[1]),
        lambda x: x.sample(),
        name="p_x")([p,s])
    return tfk.Model(z,x)

def inference_network(input_dim, latent_dim, n_layer, n_hidden, eps_dim):
    x = tfk.layers.Input(input_dim)
    eps = tfk.layers.Input(eps_dim)
    h = tfk.layers.Concatenate()([x, eps])
    h = tfk.layers.Dense(n_hidden, activation='relu')(h)
    h = tfk.layers.Dense(n_hidden, activation='relu')(h)
    z = tfk.layers.Dense(latent_dim)(h)
    return tfk.Model([x,eps],z)

def data_network(input_dim, latent_dim, n_layers=2, n_hidden=128, activation_fn=None):
    x = tfk.layers.Input(input_dim)
    z = tfk.layers.Input(latent_dim)
    if use_vanillagan:
        h = z
    else:
        h = tfk.layers.Concatenate()([x,z])
    h = tfk.layers.Dense(n_hidden, activation='relu')(h)
    log_d = tfk.layers.Dense(1, activation=activation_fn)(h)
    return tfk.Model([x,z], log_d)

# Construct model and training ops

p_model = generative_network(params['batch_size'], params['latent_dim'],
                             params['input_dim'], params['n_layer_gen'],
                             params['n_hidden_gen'])

q_model = inference_network(params['input_dim'], params['latent_dim'],
                            params['n_layer_inf'], params['n_hidden_inf'],
                            params['eps_dim'])

log_d_model = data_network(params['input_dim'], params['latent_dim'],
                           n_layers=params['n_layer_disc'],
                           n_hidden=params['n_hidden_disc'])

# Define keras models
x = tfk.layers.Input(params['input_dim'])
eps = tfk.layers.Input(params['eps_dim'])
p_z = tfk.layers.Input(params['latent_dim'])

p_x = tfk.Model(p_z, p_model(p_z))
q_z = tfk.Model([x,eps], q_model([x,eps]))
log_d = tfk.Model([x, eps, p_z], [log_d_model([p_x(p_z), p_z]),        # prior
                                  log_d_model([x, q_z([x,eps])])])     # posterior
if use_veegan or use_vanillagan:
    log_d_posterior = tfk.Model([x, eps, p_z], [log_d_model([x, q_z([x,eps])]),
                                                p_x(q_z([x,eps])).log_prob(x)])
elif use_bigan:
    log_d_posterior = tfk.Model([x, eps, p_z], [log_d_model([x, q_z([x,eps])]),
                                                log_d_model([p_x(p_z), p_z])])

# Compile: define optimizers and losses
for l in log_d_model.layers: l.trainable=True
for l in q_model.layers: l.trainable=False
for l in p_model.layers: l.trainable=False
opt = tfk.optimizers.Adam(learning_rate=params['learning_rate'], beta_1=0.5)
log_d.compile(opt, [tf.nn.sigmoid_cross_entropy_with_logits]*2)

def log_d_posterior_loss(true, pred):
    return tf.reduce_mean(pred)
def recon_likelihood(true, pred):
    return -tf.reduce_mean(tf.reduce_sum(pred,axis=1))*(1-use_vanillagan)
for l in log_d_model.layers: l.trainable=False
for l in q_model.layers: l.trainable=True
for l in p_model.layers: l.trainable=True
if use_veegan or use_vanillagan:
    log_d_posterior.compile(opt, [log_d_posterior_loss, recon_likelihood])
elif use_bigan:
    log_d_posterior.compile(opt, [tf.nn.sigmoid_cross_entropy_with_logits]*2)

ones = np.ones((params['batch_size'],1))
zeros = np.zeros((params['batch_size'],1))

# Visualization

fig, (ax,ax2) = plt.subplots(1,2, num=1, figsize=(8,4))
lims = None
frame = 0
z_input_viz = normal_mixture([500, params['latent_dim']])
x_input_viz = tf.random.normal([500, params['input_dim']])
eps_input_viz = tf.random.normal([500, params['eps_dim']])
z_scat = ax.scatter([0],[0], label='target')
qz_scat = ax.scatter([0],[0], label='generated')
def viz(epoch):
    global frame, lims
    z_input_viz = normal_mixture([500, params['latent_dim']])
    x_input_viz = tf.random.normal([500, params['input_dim']])
    eps_input_viz = tf.random.normal([500, params['eps_dim']])
    mus = np.array([np.array([i, j]) for i, j in product(range(-4, 5, 2),
                                                         range(-4, 5, 2))],
                   dtype=np.float32)
    x_output = q_z.predict([x_input_viz, eps_input_viz])
    z_output = p_x.predict(z_input_viz)
    xl_ = np.array([np.minimum(np.min(z_input_viz[:,0]), np.min(x_output[:,0])),
                    np.maximum(np.max(z_input_viz[:,0]), np.max(x_output[:,0]))])
    yl_ = np.array([np.minimum(np.min(z_input_viz[:,1]), np.min(x_output[:,1])),
                    np.maximum(np.max(z_input_viz[:,1]), np.max(x_output[:,1]))])
    if lims is None:
        lims = np.hstack([xl_, yl_])
    else:
        lims[0] = xl_[0] if xl_[0] < lims[0] else lims[0]*0.95+xl_[0]*0.05
        lims[1] = xl_[1] if xl_[1] > lims[1] else lims[1]*0.95+xl_[1]*0.05
        lims[2] = yl_[0] if yl_[0] < lims[2] else lims[2]*0.95+yl_[0]*0.05
        lims[3] = yl_[1] if yl_[1] > lims[3] else lims[3]*0.95+yl_[1]*0.05
    ax.set_xlim(lims[:2]); ax.set_ylim(lims[2:4])
    z_scat.set_offsets(z_input_viz)
    qz_scat.set_offsets(x_output)
    f = ax.set_title('data', fontname='cmr10')
    ax.legend(loc=1, prop=f.get_fontproperties())
    ax2.clear()
    # ax2.scatter(x_input_viz[:,0], x_input_viz[:,1])
    for m in mus:
        z_input_viz = np.hstack([tf.random.normal([50, 1], mean=m[0], stddev=0.05),
                                 tf.random.normal([50, 1], mean=m[1], stddev=0.05)])
        z_output = p_x.predict(z_input_viz)
        ax2.scatter(z_output[:,0], z_output[:,1])
    ax2.plot(np.cos(np.linspace(0,2*np.pi,200))*1.96,
             np.sin(np.linspace(0,2*np.pi,200))*1.96, 'k--', label='target space')
    ax2.set_xlim(-3,3)
    ax2.set_ylim(-3,3)
    if params['input_dim']==2:
        ax2.set_title('input / inferred', fontname='cmr10')
    else:
        ax2.set_title('input / inferred (first 2 dims)', fontname='cmr10')
    ax2.legend(loc=1, prop=f.get_fontproperties())
    # ax2.plot([-1,1,1,-1,-1], [-1,-1,1,1,-1], 'k--')
    fig.suptitle(f'{params["input_dim"]}D {method}: Epoch {epoch}', fontname='cmr10')
    # fig.canvas.draw()
    # plt.pause(0.0001)
    fig.savefig('now.png')
    fig.savefig(f'frames/frame{frame:06d}.png')
    frame += 1

# Training

with tqdm(range(200),total=200) as tq:
    for i in tq:
        for j in range(100):
            x_input = tf.random.normal([params['batch_size'], params['input_dim']])
            z_input = normal_mixture([params['batch_size'], params['latent_dim']])
            eps_input = tf.random.normal([params['batch_size'], params['eps_dim']])
            for l in log_d_model.layers: l.trainable=True
            for l in q_model.layers: l.trainable=False
            for l in p_model.layers: l.trainable=False
            log_d.train_on_batch([x_input,eps_input,z_input], [zeros, ones])

            x_input = tf.random.normal([params['batch_size'], params['input_dim']])
            z_input = normal_mixture([params['batch_size'], params['latent_dim']])
            eps_input = tf.random.normal([params['batch_size'], params['eps_dim']])
            for l in log_d_model.layers: l.trainable=False
            for l in q_model.layers: l.trainable=True
            for l in p_model.layers: l.trainable=True
            log_d_posterior.train_on_batch([x_input,eps_input,z_input], [zeros, ones])
        viz(i)
