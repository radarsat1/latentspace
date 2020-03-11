
# Steve's adaptation of VEEGAN example:
# https://github.com/akashgit/VEEGAN/blob/master/VEEGAN_2D_RING.ipynb
# to TensorFlow 2 + Keras, with the TensorFlow Probabilities library:
# https://blog.tensorflow.org/2019/03/variational-autoencoders-with.html
#
# Stephen Sinclair <radarsat1@gmail.com>

# Imports

import matplotlib as mpl
# mpl.use('Agg')
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from tqdm import tqdm

ds = tfp.distributions

# Parameters

params = {
    'batch_size': 500,
    'latent_dim': 2, 
    'eps_dim': 1, 
    'input_dim': 254,
    'n_layer_disc': 2,
    'n_hidden_disc': 128,
    'n_layer_gen': 2,
    'n_hidden_gen': 128,
    'n_layer_inf': 2,
    'n_hidden_inf': 128,
    'learning_rate': 1e-3,
}

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
    x = tfp.layers.DistributionLambda(
        lambda x: tfp.distributions.Normal(loc=x, scale=1),
        lambda x: x.sample(),
        name="p_x")(p)
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
log_d_posterior = tfk.Model([x, eps], [log_d_model([x, q_z([x,eps])]), # posterior
                                       p_x(q_z([x,eps])).log_prob(x)])

# Compile: define optimizers and losses
for l in log_d_model.layers: l.trainable=True
for l in q_model.layers: l.trainable=False
for l in p_model.layers: l.trainable=False
opt = tfk.optimizers.Adam(learning_rate=params['learning_rate'], beta_1=0.5)
log_d.compile(opt, [lambda t,p: tf.nn.sigmoid_cross_entropy_with_logits(t,p)]*2)

def log_d_posterior_loss(true, pred):
    return tf.reduce_mean(pred)
def recon_likelihood(true, pred):
    return -tf.reduce_mean(tf.reduce_sum(pred,axis=1))
for l in log_d_model.layers: l.trainable=False
for l in q_model.layers: l.trainable=True
for l in p_model.layers: l.trainable=True
log_d_posterior.compile(opt, [log_d_posterior_loss, recon_likelihood])

ones = np.ones((params['batch_size'],1))
zeros = np.zeros((params['batch_size'],1))

# Visualization

fig, ax = plt.subplots(1,1, num=1)
lims = None
frame = 0
z_input_viz = normal_mixture([500, params['latent_dim']])
x_input_viz = tf.random.normal([500, params['input_dim']])
eps_input_viz = tf.random.normal([500, params['eps_dim']])
z_scat = ax.scatter([0],[0])
qz_scat = ax.scatter([0],[0])
def viz(epoch):
    global frame, lims
    x_output = q_z.predict([x_input_viz, eps_input_viz])
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
    fig.suptitle(f'VEEGAN: Epoch {epoch}')
    fig.canvas.draw()
    plt.pause(0.0001)
    # fig.savefig('now.png')
    # fig.savefig(f'frames/frame{frame:06d}.png')
    frame += 1

# Training

with tqdm(range(100),total=100) as tq:
    for i in tq:
        for j in range(1000):
            x_input = tf.random.normal([params['batch_size'], params['input_dim']])
            z_input = normal_mixture([params['batch_size'], params['latent_dim']])
            eps_input = tf.random.normal([params['batch_size'], params['eps_dim']])
            for l in log_d_model.layers: l.trainable=True
            for l in q_model.layers: l.trainable=False
            for l in p_model.layers: l.trainable=False
            log_d.train_on_batch([x_input,eps_input,z_input], [zeros, ones])

            x_input = tf.random.normal([params['batch_size'], params['input_dim']])
            eps_input = tf.random.normal([params['batch_size'], params['eps_dim']])
            for l in log_d_model.layers: l.trainable=False
            for l in q_model.layers: l.trainable=True
            for l in p_model.layers: l.trainable=True
            log_d_posterior.train_on_batch([x_input,eps_input], [zeros, zeros])
        viz(i)
