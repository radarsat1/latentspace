
# Adaptation of VEEGAN example:
# https://github.com/akashgit/VEEGAN/blob/master/VEEGAN_2D_RING.ipynb
# to TensorFlow 2 + Keras, with the TensorFlow Probabilities library:
# https://blog.tensorflow.org/2019/03/variational-autoencoders-with.html
#
# Stephen Sinclair <radarsat1@gmail.com>

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability as tfp

dists = tfp.distributions

class Model(object):
    def __init__(self, params, dataset):
        self.params = params
        self.dataset = dataset

    def decoder_network(self):
        return self.inference_network(self.dataset.params['latent_dim'],
                                      self.dataset.params['data_dim'],
                                      self.params['shape']['n_layer_inf'],
                                      self.params['shape']['n_hidden_inf'],
                                      self.params['eps_dim'])

    def encoder_network(self):
        return self.generative_network(self.dataset.params['data_dim'],
                                       self.dataset.params['latent_dim'],
                                       self.params['shape']['n_layer_gen'],
                                       self.params['shape']['n_hidden_gen'])

    def critic_network(self):
        return self.data_network(self.dataset.params['data_dim'],
                                 self.dataset.params['latent_dim'],
                                 n_layers=self.params['shape']['n_layer_disc'],
                                 n_hidden=self.params['shape']['n_hidden_disc'])

    def generative_network(self, latent_dim, input_dim, n_layer,
                           n_hidden, eps=1e-6, X=None):
        Dense = tfk.layers.Dense
        if self.params['normalization']['gen'] == 'batch':
            Norm = tfk.layers.BatchNormalization
        elif self.params['normalization']['gen'] == 'layer':
            Norm = tfk.layers.LayerNormalization
        else:
            Norm = lambda: (lambda x: x)
        z = tfk.layers.Input((latent_dim,), name="p_z")
        h = Dense(n_hidden)(z)
        j = h = Norm()(h)
        h = tfk.layers.LeakyReLU()(h)
        h = Dense(n_hidden)(h)
        k = h = Norm()(h)
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
            lambda x: dists.MultivariateNormalTriL(loc=x[0], scale_tril=x[1]),
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

    def inference_network(self, input_dim, latent_dim, n_layer, n_hidden, eps_dim):
        Dense = tfk.layers.Dense
        if self.params['normalization']['gen'] == 'batch':
            Norm = tfk.layers.BatchNormalization
        elif self.params['normalization']['gen'] == 'layer':
            Norm = tfk.layers.LayerNormalization
        else:
            Norm = lambda: (lambda x: x)
        x = tfk.layers.Input(input_dim)
        eps = tfk.layers.Input(eps_dim)
        h = tfk.layers.Concatenate()([x, eps])
        h = Dense(n_hidden)(h)
        j = h = Norm()(h)
        h = tfk.layers.LeakyReLU()(h)
        h = Dense(n_hidden)(h)
        k = h = Norm()(h)
        h = tfk.layers.LeakyReLU()(h)
        # h = tfk.layers.Add()([h,j])
        # h = Dense(n_hidden)(h)
        # h = tfk.layers.BatchNormalization()(h)
        # h = tfk.layers.LeakyReLU()(h)
        # h = tfk.layers.Add()([h,k])
        z = Dense(latent_dim)(h)
        return tfk.Model([x,eps],z)

    def data_network(self, input_dim, latent_dim, n_layers=2,
                     n_hidden=128, activation_fn=None):
        if self.params['variant'] == 'sn':
            from SpectralNormalizationKeras import DenseSN as Dense
        else:
            Dense = tfk.layers.Dense
        if self.params['normalization']['critic'] == 'batch':
            Norm = tfk.layers.BatchNormalization
        elif self.params['normalization']['critic'] == 'layer':
            Norm = tfk.layers.LayerNormalization
        else:
            Norm = lambda: (lambda x: x)

        x = tfk.layers.Input(input_dim)
        z = tfk.layers.Input(latent_dim)
        if self.params['type'] == 'gan':
            h = z
        else:
            h = tfk.layers.Concatenate()([x,z])
        h = Dense(n_hidden)(h)
        j = h = Norm()(h)
        h = tfk.layers.LeakyReLU()(h)
        h = tfk.layers.Dropout(0.2)(h)
        h = Dense(n_hidden)(h)
        k = h = Norm()(h)
        h = tfk.layers.LeakyReLU()(h)
        # h = tfk.layers.Add()([h,j])
        h = tfk.layers.Dropout(0.1)(h)
        log_d = Dense(1, activation=activation_fn)(h)
        return tfk.Model([x,z], log_d)
