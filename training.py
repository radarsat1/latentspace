# Training framework for bidrectional GAN problems

# First deal wtih parameters
import sys, os

# Dataset parameters
dataset_params = {
    'name': 'multimodal_points',
    'data_dim': 2,
    'latent_dim': 2,
    'latent_prior': ['normal','uniform'][0],
}

# Model parameters
model_params = {
    'name': 'dense',
    'type': ['veegan','bigan','vanilla'][1],
    'loss': ['sigmoid','wgan','began'][1],
    'variant': ['sn','0gp','1gp'][1],
    'gp_weight': 0.01,
    'normalization': {'gen':'batch', 'critic':None},
    'eps_dim': 1,
    'shape': {
        'n_layer_disc': 2,
        'n_hidden_disc': 200,
        'n_layer_gen': 2,
        'n_hidden_gen': 200,
        'n_layer_inf': 2,
        'n_hidden_inf': 200,
    },
}

# Training parameters
training_params = {
    'batch_size': 100,
    'epochs': 5000,
    'epoch_size': 100,
    'learning_rate': 1e-3,
    'learning_rate_target': 1e-4,
    'critic_ratio': 1,
}

# Override parameters on command-line
all_params = {'dataset': dataset_params,
              'model': model_params,
              'training': training_params}
try:
    for i in range(1,len(sys.argv)):
        arg = sys.argv[i].split('=')
        p = all_params
        for a in arg[0].split('.'):
            parm = p
            p = p[a]
        if isinstance(parm[a],str):     parm[a] = arg[1]
        elif parm[a] is None:           parm[a] = arg[1]
        elif isinstance(parm[a],float): parm[a] = float(arg[1])
        elif isinstance(parm[a],int):   parm[a] = int(arg[1])
        else: assert False and 'Unknown parameter type.'
except KeyError as e:
    print(f'Unknown field {e} in {sys.argv[i]}')
    exit(1)

# Start of program

import matplotlib as mpl
mpl.use('Agg')
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif'] = ['cmr10']
# mpl.font_manager.findfont('cmmr')

import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
from tqdm import tqdm

from datasets import get_dataset
from models import get_model

# Get dataset and model

ds = get_dataset(dataset_params)
model = get_model(model_params, ds)

# Construct model and training ops

decoder = model.decoder_network()
encoder = model.encoder_network()
critic = model.critic_network()

from utils.gp import GradientPenalty

# Define keras models
z = tfk.layers.Input(dataset_params['latent_dim'])
eps = tfk.layers.Input(model_params['eps_dim'])
p_x = tfk.layers.Input(dataset_params['data_dim'])

p_z = tfk.Model(p_x, encoder(p_x))
q_x = tfk.Model([z,eps], decoder([z,eps]))

# We train the data model to discriminate between the prior and
# posterior distributions.
data_model = tfk.Model([p_x, eps, z],
                       [critic([p_x, p_z(p_x)]),        # prior
                        critic([q_x([z,eps]), z]),      # posterior
                        GradientPenalty(model_params, critic)(
                            [p_x, p_z(p_x)],
                            [q_x([z,eps]), z])])

# We train the generator model to produce posterior distributions that
# are hard to discriminate.
if model_params['type']=='veegan':
    gen_model = tfk.Model([p_x, eps, z], [critic([q_x([z,eps]), z]),
                                          tf.reshape(p_z(q_x([z,eps])).log_prob(z),
                                                     (-1,1))])
elif model_params['type'] in ['bigan','vanilla']:
    gen_model = tfk.Model([p_x, eps, z], [critic([q_x([z,eps]), z]),
                                          critic([p_x, p_z(p_x)])])
else:
    assert False and 'Unknown model type.'

wgan_loss = lambda t,p: tf.reduce_mean(t*p)
dummy_gp_loss = lambda t,p: tf.reduce_mean(p)*(model_params['variant'] in ['0gp','1gp'])

# Compile: define optimizers and losses
for l in critic.layers: l.trainable=True
for l in encoder.layers: l.trainable=False
for l in decoder.layers: l.trainable=False
opt = tfk.optimizers.Adam(learning_rate=training_params['learning_rate'],
                          beta_1=0.5)
if model_params['loss'] == 'wgan':
    data_model.compile(opt, [wgan_loss]*2+[dummy_gp_loss])
elif model_params['loss'] == 'sigmoid':
    data_model.compile(opt, [tf.nn.sigmoid_cross_entropy_with_logits]*2+[dummy_gp_loss])
# data_model.compile(opt, ['binary_crossentropy']*2)

def gen_model_loss(true, pred):
    return tf.reduce_mean(pred)
def recon_likelihood(true, pred):
    return -tf.reduce_mean(tf.reduce_sum(pred,axis=1))*(model_params['type']=='veegan')
for l in critic.layers: l.trainable=False
for l in encoder.layers: l.trainable=True
for l in decoder.layers: l.trainable=True
opt = tfk.optimizers.Adam(learning_rate=training_params['learning_rate'], beta_1=0.5)
if model_params['type'] in ['veegan','vanilla']:
    if model_params['loss'] == 'wgan':
        gen_model.compile(opt, [wgan_loss, recon_likelihood])
    elif model_params['loss'] == 'sigmoid':
        gen_model.compile(opt, [gen_model_loss, recon_likelihood])
elif model_params['type'] == 'bigan':
    if model_params['loss'] == 'wgan':
        gen_model.compile(opt, [wgan_loss]*2)
    elif model_params['loss'] == 'sigmoid':
        gen_model.compile(opt, [tf.nn.sigmoid_cross_entropy_with_logits]*2)
    # gen_model.compile(opt, ['binary_crossentropy']*2)
else:
    assert False and 'Not a known model'

reals = np.ones((training_params['batch_size'],1))
if model_params['loss'] == 'wgan':
    fakes = -np.ones((training_params['batch_size'],1))
else:
    fakes = np.zeros((training_params['batch_size'],1))
dummy = reals

# Training

learning_rate = training_params['learning_rate']
rate_decay = np.exp((np.log(training_params['learning_rate_target'])
                     - np.log(learning_rate))
                     / (training_params['epochs'] * training_params['epoch_size']))

z_input_gen = ds.latent_generator(training_params['batch_size'])
x_input_gen = ds.data_generator(training_params['batch_size'])
def eps_input_generator():
    while True:
        yield tf.random.normal([training_params['batch_size'],
                                model_params['eps_dim']])
eps_input_gen = eps_input_generator()

method = {'veegan':'VEEGAN', 'bigan':'BiGAN', 'vanilla':'Vanilla GAN'}\
         [model_params['type']]
variant = model_params['variant'].upper()
if 'GP' in variant:
    variant += f' ($\\lambda=${model_params["gp_weight"]:0.2g})'

dirname = f'frames-{method}{dataset_params["data_dim"]}D-{variant}'
normgen = {'batch':'B','layer':'L'}.get(model_params['normalization']['gen'],'0')
normcritic = {'batch':'B','layer':'L'}.get(model_params['normalization']['critic'],'0')
dirname += f'-{normgen}{normcritic}{training_params["discriminator_ratio"]}'

ds.init_viz(dirname, method, variant,
            next(x_input_gen), next(z_input_gen), next(eps_input_gen))

with tqdm(range(training_params['epochs']),
          total=training_params['epochs']) as tq:
    for i in tq:
        ds.viz(i, decoder, encoder)
        for j in range(training_params['epoch_size']):
            for l in critic.layers: l.trainable=True
            for l in encoder.layers: l.trainable=False
            for l in decoder.layers: l.trainable=False
            for k in range(training_params['critic_ratio']):
                x_input = next(x_input_gen)
                z_input = next(z_input_gen)
                eps_input = next(eps_input_gen)
                data_model.train_on_batch([x_input,eps_input,z_input],
                                          [fakes, reals, dummy])

            x_input = next(x_input_gen)
            z_input = next(z_input_gen)
            eps_input = next(eps_input_gen)
            for l in critic.layers: l.trainable=False
            for l in encoder.layers: l.trainable=True
            for l in decoder.layers: l.trainable=True
            gen_model.train_on_batch([x_input,eps_input,z_input],
                                     [fakes, reals])

            learning_rate *= rate_decay
            data_model.optimizer.learning_rate.assign(learning_rate)
            gen_model.optimizer.learning_rate.assign(learning_rate)
            tq.set_postfix({'lr': learning_rate})
    ds.viz(i+1, decoder, encoder)
