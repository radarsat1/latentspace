
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

CODE=64

class Model(object):
    def __init__(self, params, dataset):
        self.params = params
        self.dataset = dataset

    def decoder_network(self, bn=True):
        L = self.dataset.params['latent_dim']
        D = self.dataset.params['data_dim']
        E = self.params['eps_dim']
        F = self.params['shape']['filters']
        if self.params['normalization']['gen'] == 'batch':
            Norm = tfk.layers.BatchNormalization
            Norm = tf.compat.v1.keras.layers.BatchNormalization
            # why? https://github.com/tensorflow/tensorflow/issues/37673
        elif self.params['normalization']['gen'] == 'layer':
            Norm = tfk.layers.LayerNormalization
        else:
            Norm = lambda: (lambda x: x)
        if not bn: Norm = lambda: (lambda x: x)

        inp = x = tfkl.Input(L)
        e = tfkl.Input(E)
        # xi = tfkl.Reshape((self.dataset.params['latent_dim'],1))(x)
        # xe = tfkl.Reshape((self.dataset.params['latent_dim'],1))(e)

        # x = tfkl.Concatenate()([x,e])

        # x = tfkl.Dense(L*F)(x)
        # x = tfkl.Reshape((L,F))(x)
        x = tfkl.Dense(CODE)(x)
        x = tfkl.LeakyReLU()(x)
        x = tfkl.Reshape((CODE,1))(x)
        x = tfkl.Conv1D(F,1,padding='causal')(x)
        # x = Norm()(x)
        x = tfkl.LeakyReLU()(x)
        while x.shape[1] < D:
            y = x
            x = tfkl.Conv1D(F,7,padding='causal')(x)
            x = Norm()(x)
            x = tfkl.LeakyReLU()(x)
            x = tfkl.Add()([x,y])
            x = tfkl.UpSampling1D(4)(x)
        x = tfkl.Conv1D(F,5,padding='causal')(x)
        x = Norm()(x)
        x = tfkl.LeakyReLU()(x)
        x = tfkl.Conv1D(1,5,padding='causal')(x)
        x = tfkl.Reshape((D,))(x)
        # x = tfkl.Activation('tanh')(x)
        return tfk.Model([inp,e],x)

    def encoder_network(self, bn=None,
                        stochastic=True, endit=True, extradim=False):
        L = self.dataset.params['latent_dim']
        TL = L + sum(range(L+1))
        D = self.dataset.params['data_dim']
        F = self.params['shape']['filters']
        if bn is None:
            bn = self.params['normalization']['gen']
        if bn == 'batch':
            Norm = tfk.layers.BatchNormalization
            Norm = tf.compat.v1.keras.layers.BatchNormalization
            # why? https://github.com/tensorflow/tensorflow/issues/37673
        elif bn == 'layer':
            Norm = tfk.layers.LayerNormalization
        else:
            Norm = lambda: (lambda x: x)
        if not bn: Norm = lambda: (lambda x: x)
        if extradim:
            inp = x = tfkl.Input([D,extradim])
        else:
            inp = x = tfkl.Input(D)
            x = tfkl.Reshape((D,1))(x)
        x = tfkl.Conv1D(F,3,padding='same')(x)
        x = tfkl.LeakyReLU()(x)
        while x.shape[1] > TL*0+CODE:
            y = x
            x = tfkl.Conv1D(F,7,padding='same')(x)
            x = Norm()(x)
            x = tfkl.LeakyReLU()(x)
            x = tfkl.Add()([x,y])
            # x = tfkl.Conv1D(F,7,padding='same',strides=4)(x)
            x = tfkl.AvgPool1D(4)(x)
        x = Norm()(x)
        x = tfkl.LeakyReLU()(x)
        x = tfkl.Conv1D(1,3,padding='same')(x)
        x = Norm()(x)
        x = tfkl.LeakyReLU()(x)
        x = tfkl.Flatten()(x)

        x = tfkl.Dense(CODE)(x)
        x = Norm()(x)
        x = tfkl.LeakyReLU()(x)

        if stochastic and endit:
            p = tfkl.Dense(TL)(x)
            s = tfp.bijectors.FillScaleTriL().forward(p[:,L:])
            p = p[:,:L]
            x = tfp.layers.DistributionLambda(
                lambda x: dists.MultivariateNormalTriL(loc=x[0], scale_tril=x[1]),
                lambda x: x.sample(),
                name="p_x")([p,s])
        elif endit:
            x = tfkl.Dense(L)(x)

        return tfk.Model(inp,x)

    def critic_network(self):
        # TODO: critic batch/layer normalization
        # TODO: spectral normalization
        D = self.dataset.params['data_dim']
        L = self.dataset.params['latent_dim']
        F = self.params['shape']['filters']
        inp = x = tfkl.Input(D)
        x = tfkl.Dropout(0.3)(x)
        x = tfkl.Reshape((D,1))(x)
        lat = z = tfkl.Input(L)
        bn = False
        if self.params['normalization']['critic'] is not None:
            bn = self.params['normalization']['critic']
        ed = 1
        if True and self.params['type'] != 'gan':
            z = tfkl.Reshape((1,L))(z)
            z = tfkl.Lambda(lambda y: tf.repeat(y, D, axis=1))(z)
            x = tfkl.Concatenate()([x,z])
            ed += L
        x = self.encoder_network(bn=bn,stochastic=False,endit=False,extradim=ed)(x)
        if False and self.params['type'] != 'gan':
            x = tfkl.Concatenate()([x,z])
        x = tfkl.Dense(L)(x)
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
