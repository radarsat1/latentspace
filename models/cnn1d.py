
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
tfkl = tfk.layers

class Model(object):
    def __init__(self, params, dataset):
        self.params = params
        self.dataset = dataset

    def decoder_network(self, bn=True):
        if self.params['normalization']['gen'] == 'batch':
            Norm = tfk.layers.BatchNormalization
            Norm = tf.compat.v1.keras.layers.BatchNormalization
            # why? https://github.com/tensorflow/tensorflow/issues/37673
        elif self.params['normalization']['gen'] == 'layer':
            Norm = tfk.layers.LayerNormalization
        else:
            Norm = lambda: (lambda x: x)
        if not bn: Norm = lambda: (lambda x: x)
        inp = x = tfkl.Input(self.dataset.params['latent_dim'])
        e = tfkl.Input(self.params['eps_dim'])
        # xi = tfkl.Reshape((self.dataset.params['latent_dim'],1))(x)
        # xe = tfkl.Reshape((self.dataset.params['latent_dim'],1))(e)
        x = tfkl.Concatenate()([x,e])
        x = tfkl.Dense(self.dataset.params['latent_dim']*self.params['filters'])(x)
        x = tfkl.LeakyReLU()(x)
        x = Norm()(x)
        x = tfkl.Reshape((self.dataset.params['latent_dim'],self.params['filters']))(x)
        while x.shape[1] < self.dataset.params['data_dim']:
            y = x
            x = tfkl.Conv1D(self.params['filters'],3,padding='same')(x)
            x = tfkl.LeakyReLU()(x)
            x = Norm()(x)
            x = tfkl.Add()([x,y])
            x = tfkl.UpSampling1D(2)(x)
        x = tfkl.Conv1D(self.params['filters'],3,padding='same')(x)
        x = tfkl.LeakyReLU()(x)
        x = Norm()(x)
        x = tfkl.Conv1D(1,3,padding='same')(x)
        x = tfkl.Reshape((self.dataset.params['data_dim'],))(x)
        return tfk.Model([inp,e],x)

    def encoder_network(self, bn=True, stochastic=True, extradim=False):
        if self.params['normalization']['gen'] == 'batch':
            Norm = tfk.layers.BatchNormalization
            Norm = tf.compat.v1.keras.layers.BatchNormalization
            # why? https://github.com/tensorflow/tensorflow/issues/37673
        elif self.params['normalization']['gen'] == 'layer':
            Norm = tfk.layers.LayerNormalization
        else:
            Norm = lambda: (lambda x: x)
        if not bn: Norm = lambda: (lambda x: x)
        if extradim:
            inp = x = tfkl.Input([self.dataset.params['data_dim'],extradim])
        else:
            inp = x = tfkl.Input(self.dataset.params['data_dim'])
            x = tfkl.Reshape((self.dataset.params['data_dim'],1))(x)
        L = self.dataset.params['latent_dim']
        TL = L + sum(range(L+1))
        x = tfkl.Conv1D(self.params['filters'],3,padding='same')(x)
        while x.shape[1] > TL:
            y = x
            x = tfkl.Conv1D(self.params['filters'],3,padding='same')(x)
            x = tfkl.LeakyReLU()(x)
            x = Norm()(x)
            x = tfkl.Add()([x,y])
            x = tfkl.MaxPool1D(2)(x)
        x = tfkl.Conv1D(self.params['filters'],3,padding='same')(x)
        x = tfkl.LeakyReLU()(x)
        x = Norm()(x)
        x = tfkl.Conv1D(1,1)(x)
        x = tfkl.Flatten()(x)
        x = tfkl.LeakyReLU()(x)
        x = Norm()(x)

        if stochastic:
            p = tfkl.Dense(TL)(x)
            s = tfp.bijectors.FillScaleTriL().forward(p[:,L:])
            p = p[:,:L]
            x = tfp.layers.DistributionLambda(
                lambda x: dists.MultivariateNormalTriL(loc=x[0], scale_tril=x[1]),
                lambda x: x.sample(),
                name="p_x")([p,s])
        else:
            p = tfkl.Dense(L)(x)

        return tfk.Model(inp,x)

    def critic_network(self):
        # TODO: critic batch/layer normalization
        # TODO: spectral normalization
        Norm = tf.compat.v1.keras.layers.BatchNormalization
        D = self.dataset.params['data_dim']
        L = self.dataset.params['latent_dim']
        inp = x = tfkl.Input(D)
        x = tfkl.Reshape((D,1))(x)
        lat = z = tfkl.Input(L)
        z = tfkl.Reshape((1,L))(z)
        z = tfkl.Lambda(lambda y: tf.repeat(y, D, axis=1))(z)
        x = tfkl.Concatenate()([x,z])
        x = self.encoder_network(bn=False,stochastic=False,extradim=1+L)(x)
        x = tfkl.Dense(L*self.params['filters'])(x)
        x = tfkl.LeakyReLU()(x)
        x = tfkl.Dense(1)(x)
        return tfk.Model([inp,lat],x)

if __name__=='__main__':
    class Tester:
        def __init__(self):
            self.params = {'data_dim': 8, 'latent_dim': 2}
    m = Model({'eps_dim':2, 'filters':10,
               'normalization': {'gen': 'batch'}}, Tester())
    x = tf.constant([[0.0,1,2,3,4,5,6,7]])
    e = tf.constant([[0.0,0]])
    y = tf.constant([[0.0,1]])
    print(m.decoder_network()([y,e]))
    print(m.encoder_network()(x))
    print(m.critic_network()([x,y]))
