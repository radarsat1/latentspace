
# Steve's adaptation of VEEGAN example:
# https://github.com/akashgit/VEEGAN/blob/master/VEEGAN_2D_RING.ipynb
# to TensorFlow 2 + Keras, with the TensorFlow Probabilities library:
# https://blog.tensorflow.org/2019/03/variational-autoencoders-with.html
#
# Stephen Sinclair <radarsat1@gmail.com>

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
import sys, os
Dense = tfk.layers.Dense

ds = tfp.distributions

use_veegan = False
use_bigan = True
use_vanillagan = False

use_veegan = sys.argv[1]=='1'
use_bigan = sys.argv[1]=='2'
use_vanillagan = sys.argv[1]=='3'

use_norm = True
use_norm_disc = True
use_wgan = True
use_wgan_sn = False
use_wgan_0gp = True
use_wgan_1gp = False
gp_lambda = 0.01

# Parameters

# Note: potentially confusing; what VEEGAN calls "latent" is actually
# the data space, and the "latent" space is "input_dim".  I kept this
# from the VEEGAN example code but it's a bit weird, I believe it's
# because they consider the structure of the GAN as an autoencoder
# from input x to p(x|z), where the latent space of that autoencoder,
# q(z|x), would be the data; this is the reverse of a BiGAN which
# would decode data p(x|z) from input z and infer q(z|x) from p(x).
params = {
    'batch_size': 100,
    'epochs': 5000,
    'epoch_size': 100,
    'learning_rate': 1e-4,
    'learning_rate_target': 1e-4,
    'discriminator_ratio': 1,
    'latent_dim': 2, # actually data dims!
    'eps_dim': 1, 
    'input_dim': 2, #254, # actually latent dims!
    'n_layer_disc': 2,
    'n_hidden_disc': 200,
    'n_layer_gen': 2,
    'n_hidden_gen': 200,
    'n_layer_inf': 2,
    'n_hidden_inf': 200,
}

if use_veegan: method = 'VEEGAN'
elif use_bigan: method = 'BiGAN'
elif use_vanillagan: method = 'Vanilla GAN'

use_norm = sys.argv[2]=='1'
use_norm_disc = sys.argv[3]=='1'
params['discriminator_ratio'] = int(sys.argv[4])

gan_type = 'GAN'
if use_wgan: gan_type = 'WGAN'
if use_wgan_sn: gan_type = 'WGAN-SN'
if use_wgan_0gp: gan_type = 'WGAN-0GP'
if use_wgan_1gp: gan_type = 'WGAN-1GP'

# Generate 2D grid

mus = np.array([np.array([i, j]) for i, j in product(range(-4, 5, 2),
                                                     range(-4, 5, 2))],
               dtype=np.float32)
# mus = np.array([np.array([i, j]) for i, j in product(range(-1, 1, 1),
#                                                      range(-1, 1, 1))],
#                dtype=np.float32)

@tf.function
def create_distribution(batch_size, num_components=25, num_features=2,**kwargs):
    num_components = len(mus)
    cat = ds.Categorical(tf.zeros(num_components, dtype=np.float32))

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
    j = h = Dense(n_hidden)(z)
    if use_norm:
        j = h = tfk.layers.BatchNormalization()(h)
    h = tfk.layers.LeakyReLU()(h)
    k = h = Dense(n_hidden)(h)
    if use_norm:
        k = h = tfk.layers.BatchNormalization()(h)
    h = tfk.layers.LeakyReLU()(h)
    # h = tfk.layers.Add()([h,j])
    # h = Dense(n_hidden)(h)
    # h = tfk.layers.BatchNormalization()(h)
    # h = tfk.layers.LeakyReLU()(h)
    # h = tfk.layers.Add()([h,k])

    # p = Dense(input_dim)(h)
    # x = p

    p = Dense(input_dim + sum(range(input_dim+1)))(h)
    # x = p
    s = tfp.bijectors.FillScaleTriL().forward(p[:,input_dim:])
    p = p[:,:input_dim]
    x = tfp.layers.DistributionLambda(
        lambda x: tfp.distributions.MultivariateNormalTriL(loc=x[0],
                                                           scale_tril=x[1]),
        lambda x: x.sample(),
        name="p_x")([p,s])

    # p = Dense(input_dim*2)(h)
    # s = p[:,input_dim:]
    # p = p[:,:input_dim]
    # x = tfp.layers.DistributionLambda(
    #     lambda x: tfp.distributions.Normal(loc=x[0], scale=x[1]),
    #     lambda x: x.sample(),
    #     name="p_x")([p,tf.nn.softplus(s)])

    # p = Dense(tfp.layers.MultivariateNormalTriL.params_size(input_dim))(h)
    # x = tfp.layers.MultivariateNormalTriL(input_dim)(p)
    return tfk.Model(z,x)

def inference_network(input_dim, latent_dim, n_layer, n_hidden, eps_dim):
    x = tfk.layers.Input(input_dim)
    eps = tfk.layers.Input(eps_dim)
    h = tfk.layers.Concatenate()([x, eps])
    j = h = Dense(n_hidden)(h)
    if use_norm:
        j = h = tfk.layers.BatchNormalization()(h)
    h = tfk.layers.LeakyReLU()(h)
    k = h = Dense(n_hidden)(h)
    if use_norm:
        k = h = tfk.layers.BatchNormalization()(h)
    h = tfk.layers.LeakyReLU()(h)
    # h = tfk.layers.Add()([h,j])
    # h = Dense(n_hidden)(h)
    # h = tfk.layers.BatchNormalization()(h)
    # h = tfk.layers.LeakyReLU()(h)
    # h = tfk.layers.Add()([h,k])
    z = Dense(latent_dim)(h)
    return tfk.Model([x,eps],z)

if use_wgan_sn:
    from SpectralNormalizationKeras import DenseSN as Dense

def random_weighted_average(x):
    true,fake = x
    weights = tf.random.uniform((tf.shape(x)[0], 1))
    return (weights * true) + ((1 - weights) * fake)

def data_network(input_dim, latent_dim, n_layers=2, n_hidden=128, activation_fn=None):
    x = tfk.layers.Input(input_dim)
    z = tfk.layers.Input(latent_dim)
    if use_vanillagan:
        h = z
    else:
        h = tfk.layers.Concatenate()([x,z])
    j = h = Dense(n_hidden)(h)
    if use_norm_disc:
        j = h = tfk.layers.BatchNormalization()(h)
    h = tfk.layers.LeakyReLU()(h)
    h = tfk.layers.Dropout(0.2)(h)
    k = h = Dense(n_hidden)(h)
    if use_norm_disc:
        k = h = tfk.layers.BatchNormalization()(h)
    h = tfk.layers.LeakyReLU()(h)
    # h = tfk.layers.Add()([h,j])
    h = tfk.layers.Dropout(0.1)(h)
    log_d = Dense(1, activation=activation_fn)(h)
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

class GradientPenalty(tfk.layers.Layer):
    def __init__(self):
        super(GradientPenalty, self).__init__()
    def call(self, x1, x2):
        if use_wgan_0gp: targ = 0
        elif use_wgan_1gp: targ = 1
        else: return tf.constant([0.0])
        r = tf.random.uniform([tf.shape(x1[0])[0],1])
        with tf.GradientTape() as tape:
            x = [x1[0]*r + x2[0]*(1-r),
                 x1[1]*r + x2[1]*(1-r)]
            loss = log_d_model(x)
        weights = [w for l in log_d_model.layers for w in l.weights if len(l.weights)>0]
        # print('loss',loss)
        grads = tape.gradient(loss, weights)[0]
        gradsl2 = tf.sqrt(tf.reduce_sum(grads**2, axis=1))
        return tf.reshape(tf.square(targ - tf.reduce_mean(gradsl2)),(1,1))*gp_lambda
#gradient_penalty = tfk.layers.Lambda(gradient_penalty)

# Define keras models
x = tfk.layers.Input(params['input_dim'])
eps = tfk.layers.Input(params['eps_dim'])
p_z = tfk.layers.Input(params['latent_dim'])

p_x = tfk.Model(p_z, p_model(p_z))
q_z = tfk.Model([x,eps], q_model([x,eps]))
log_d = tfk.Model([x, eps, p_z], [log_d_model([p_x(p_z), p_z]),        # prior
                                  log_d_model([x, q_z([x,eps])]),      # posterior
                                  GradientPenalty()([p_x(p_z), p_z],
                                                    [x, q_z([x,eps])])])
if use_veegan:
    log_d_posterior = tfk.Model([x, eps, p_z], [log_d_model([x, q_z([x,eps])]),
                                                tf.reshape(p_x(q_z([x,eps])).log_prob(x),
                                                           (-1,1))])
elif use_bigan or use_vanillagan:
    log_d_posterior = tfk.Model([x, eps, p_z], [log_d_model([x, q_z([x,eps])]),
                                                log_d_model([p_x(p_z), p_z])])

wgan_loss = lambda t,p: tf.reduce_mean(t*p)
dummy_gp_loss = lambda t,p: tf.reduce_mean(p)*(use_wgan_0gp or use_wgan_1gp)

# Compile: define optimizers and losses
for l in log_d_model.layers: l.trainable=True
for l in q_model.layers: l.trainable=False
for l in p_model.layers: l.trainable=False
opt = tfk.optimizers.Adam(learning_rate=params['learning_rate'], beta_1=0.5)
if use_wgan:
    log_d.compile(opt, [wgan_loss]*2+[dummy_gp_loss])
else:
    log_d.compile(opt, [tf.nn.sigmoid_cross_entropy_with_logits]*2+[dummy_gp_loss])
# log_d.compile(opt, ['binary_crossentropy']*2)

def log_d_posterior_loss(true, pred):
    return tf.reduce_mean(pred)
def recon_likelihood(true, pred):
    return -tf.reduce_mean(tf.reduce_sum(pred,axis=1))*use_veegan
for l in log_d_model.layers: l.trainable=False
for l in q_model.layers: l.trainable=True
for l in p_model.layers: l.trainable=True
opt = tfk.optimizers.Adam(learning_rate=params['learning_rate'], beta_1=0.5)
if use_veegan or use_vanillagan:
    if use_wgan:
        log_d_posterior.compile(opt, [wgan_loss, recon_likelihood])
    else:
        log_d_posterior.compile(opt, [log_d_posterior_loss, recon_likelihood])
elif use_bigan:
    if use_wgan:
        log_d_posterior.compile(opt, [wgan_loss]*2)
    else:
        log_d_posterior.compile(opt, [tf.nn.sigmoid_cross_entropy_with_logits]*2)
    # log_d_posterior.compile(opt, ['binary_crossentropy']*2)

ones = np.ones((params['batch_size'],1))
if use_wgan:
    zeros = -np.ones((params['batch_size'],1))
else:
    zeros = np.zeros((params['batch_size'],1))
dummy = ones

# Visualization

#fig, ((ax,ax2),(ax3,ax4)) = plt.subplots(2,2, num=1, figsize=(8,4))
fig, (ax,ax2,ax3) = plt.subplots(1,3, num=1, figsize=(9,4))
lims = None
frame = 0
z_input_viz = normal_mixture([500, params['latent_dim']])
x_input_viz = tf.random.uniform([500, params['input_dim']])
eps_input_viz = tf.random.normal([500, params['eps_dim']])
z_scat = ax.scatter([0],[0], label='target', marker='.', alpha=0.05, edgecolors='none')
qz_scat = ax.scatter([0],[0], label='generated', marker='.', alpha=0.2, edgecolors='none')
history_mus_top = []
history_sigmas_top = []
history_mus_bottom = []
history_sigmas_bottom = []
dirname = f'frames-{method}{params["input_dim"]}D-{gan_type}'
dirname += f'-{[0,1][use_norm]}{[0,1][use_norm_disc]}{params["discriminator_ratio"]}'
if not os.path.exists(dirname):
    os.mkdir(dirname)
x_input_viz = tf.random.uniform([500, params['input_dim']])
eps_input_viz = tf.random.normal([500, params['eps_dim']])
z_input_viz = np.array([np.hstack([tf.random.normal([150, 1], mean=m[0], stddev=0.05),
                                   tf.random.normal([150, 1], mean=m[1], stddev=0.05)])
                        for m in mus])
plt.subplots_adjust()
def viz(epoch):
    global frame, lims
    tf.random.set_seed(2)
    # z_input_viz = normal_mixture([500, params['latent_dim']])
    # x_input_viz = tf.random.normal([500, params['input_dim']])
    # eps_input_viz = tf.random.normal([500, params['eps_dim']])
    x_output = q_z.predict([x_input_viz, eps_input_viz])
    z_output = p_x.predict(z_input_viz.reshape((len(mus)*150,2)))
    xl_ = np.array([np.minimum(np.min(z_input_viz[:,:,0]), np.min(x_output[:,0])),
                    np.maximum(np.max(z_input_viz[:,:,0]), np.max(x_output[:,0]))])
    yl_ = np.array([np.minimum(np.min(z_input_viz[:,:,1]), np.min(x_output[:,1])),
                    np.maximum(np.max(z_input_viz[:,:,1]), np.max(x_output[:,1]))])
    if lims is None:
        lims = np.hstack([xl_, yl_])
    else:
        # lims[0] = xl_[0] if xl_[0] < lims[0] else lims[0]*0.95+xl_[0]*0.05
        # lims[1] = xl_[1] if xl_[1] > lims[1] else lims[1]*0.95+xl_[1]*0.05
        # lims[2] = yl_[0] if yl_[0] < lims[2] else lims[2]*0.95+yl_[0]*0.05
        # lims[3] = yl_[1] if yl_[1] > lims[3] else lims[3]*0.95+yl_[1]*0.05
        lims[0] = lims[0]*0.95+xl_[0]*0.05
        lims[1] = lims[1]*0.95+xl_[1]*0.05
        lims[2] = lims[2]*0.95+yl_[0]*0.05
        lims[3] = lims[3]*0.95+yl_[1]*0.05
    ax.set_xlim(lims[:2]); ax.set_ylim(lims[2:4])
    z_scat.set_offsets(z_input_viz.reshape((len(mus)*150,2)))
    qz_scat.set_offsets(x_output)
    f = ax.set_title('decoded', fontname='cmr10')
    #ax.legend(loc=1, prop=f.get_fontproperties())
    ax2.clear()
    ax3.clear()
    # ax2.scatter(x_input_viz[:,0], x_input_viz[:,1])
    all_mus = []
    all_sigmas = []
    for i,m in enumerate(mus):
        z_output = p_model.predict(z_input_viz[i])
        # z_output, mu, sigma = p_model.predict(z_input_viz)
        all_mus.append(np.mean(z_output))
        all_sigmas.append(np.std(z_output))
        ax2.scatter(z_output[:,0], z_output[:,1], marker='.', alpha=0.2, edgecolors='none')
        x_output = q_z.predict([z_output, eps_input_viz[:z_output.shape[0]]])
        ax3.scatter(x_output[:,0], x_output[:,1], marker='.', alpha=0.2, edgecolors='none')
    ax3.set_xlim(lims[:2]); ax3.set_ylim(lims[2:4])
    ax3.set_title('reconstruction', fontname='cmr10')
    history_mus_top.append(np.percentile(all_mus,90))
    history_sigmas_top.append(np.percentile(all_sigmas,90))
    history_mus_bottom.append(np.percentile(all_mus,10))
    history_sigmas_bottom.append(np.percentile(all_sigmas,10))
    # ax2.plot(np.cos(np.linspace(0,2*np.pi,200))*1.96,
    #          np.sin(np.linspace(0,2*np.pi,200))*1.96, 'k--', label='target space')
    # ax2.set_xlim(-3,3)
    # ax2.set_ylim(-3,3)
    ax2.set_xlim(-0.5,1.5)
    ax2.set_ylim(-0.5,1.5)
    if params['input_dim']==2:
        ax2.set_title('encoded', fontname='cmr10')
    else:
        ax2.set_title('encoded (first 2 dims)', fontname='cmr10')
    # ax2.legend(loc=1, prop=f.get_fontproperties())
    # ax2.plot([-1,1,1,-1,-1], [-1,-1,1,1,-1], 'k--')
    # ax3.clear()
    # x = np.arange(frame+1)
    # ax3.fill_between(x, history_mus_top, history_mus_bottom, label='mu', alpha=0.7)
    # ax3.fill_between(x, history_sigmas_top, history_sigmas_bottom, label='sigma', alpha=0.7)
    # ax3.legend(loc=2)
    fig.suptitle(f'{params["input_dim"]}D {method} {gan_type}: Epoch {epoch}', fontname='cmr10')
    # fig.canvas.draw()
    # plt.pause(0.0001)
    fig.savefig('now.png')
    fig.savefig(f'{dirname}/frame{frame:06d}.png')
    frame += 1
    tf.random.set_seed(frame)

# Training

learning_rate = params['learning_rate']
rate_decay = np.exp((np.log(params['learning_rate_target']) - np.log(learning_rate))
                     / (params['epochs'] * params['epoch_size']))
with tqdm(range(params['epochs']),total=params['epochs']) as tq:
    for i in tq:
        viz(i)
        for j in range(params['epoch_size']):
            for l in log_d_model.layers: l.trainable=True
            for l in q_model.layers: l.trainable=False
            for l in p_model.layers: l.trainable=False
            for k in range(params['discriminator_ratio']):
                x_input = tf.random.uniform([params['batch_size'], params['input_dim']])
                z_input = normal_mixture([params['batch_size'], params['latent_dim']])
                eps_input = tf.random.normal([params['batch_size'], params['eps_dim']])
                log_d.train_on_batch([x_input,eps_input,z_input], [zeros, ones, dummy])

            x_input = tf.random.uniform([params['batch_size'], params['input_dim']])
            z_input = normal_mixture([params['batch_size'], params['latent_dim']])
            eps_input = tf.random.normal([params['batch_size'], params['eps_dim']])
            for l in log_d_model.layers: l.trainable=False
            for l in q_model.layers: l.trainable=True
            for l in p_model.layers: l.trainable=True
            log_d_posterior.train_on_batch([x_input,eps_input,z_input], [zeros, ones])

            learning_rate *= rate_decay
            log_d.optimizer.learning_rate.assign(learning_rate)
            log_d_posterior.optimizer.learning_rate.assign(learning_rate)
            tq.set_postfix({'lr': learning_rate})
    viz(i+1)
