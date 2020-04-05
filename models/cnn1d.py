
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

    def decoder_network(self):
        inp = x = tfkl.Input(self.dataset.params['latent_dim'])
        e = tfkl.Input(self.params['eps_dim'])
        xi = tfkl.Reshape((self.dataset.params['latent_dim'],1))(x)
        xe = tfkl.Reshape((self.dataset.params['latent_dim'],1))(e)
        x = tfkl.Concatenate()([xi,xe])
        while x.shape[1] < self.dataset.params['data_dim']:
            x = tfkl.Conv1D(10,3,padding='same')(x)
            x = tfkl.LeakyReLU()(x)
            x = tfkl.UpSampling1D(2)(x)
        x = tfkl.Conv1D(10,3,padding='same')(x)
        x = tfkl.LeakyReLU()(x)
        x = tfkl.Conv1D(1,1)(x)
        x = tfkl.Reshape((self.dataset.params['data_dim'],))(x)
        return tfk.Model([inp,e],x)

    def encoder_network(self):
        inp = x = tfkl.Input(self.dataset.params['data_dim'])
        x = tfkl.Reshape((self.dataset.params['data_dim'],1))(x)
        L = self.dataset.params['latent_dim']
        while x.shape[1] > L:
            x = tfkl.Conv1D(10,3,padding='same')(x)
            x = tfkl.LeakyReLU()(x)
            x = tfkl.AvgPool1D(2)(x)
        x = tfkl.Conv1D(10,3,padding='same')(x)
        x = tfkl.LeakyReLU()(x)
        x = tfkl.Conv1D(1,1)(x)
        x = tfkl.Reshape((L,))(x)

        p = tfkl.Dense(L + sum(range(L+1)))(x)
        s = tfp.bijectors.FillScaleTriL().forward(p[:,L:])
        p = p[:,:L]
        x = tfp.layers.DistributionLambda(
            lambda x: dists.MultivariateNormalTriL(loc=x[0], scale_tril=x[1]),
            lambda x: x.sample(),
            name="p_x")([p,s])

        return tfk.Model(inp,x)

    def critic_network(self):
        inp = x = tfkl.Input(self.dataset.params['data_dim'])
        lat = tfkl.Input(self.dataset.params['latent_dim'])
        x = self.encoder_network()(x)
        x = tfkl.Concatenate()([x,lat])
        x = tfkl.Dense(self.dataset.params['latent_dim']*10)(x)
        x = tfkl.LeakyReLU()(x)
        x = tfkl.Dense(1)(x)
        return tfk.Model([inp,lat],x)

if __name__=='__main__':
    class Tester:
        def __init__(self):
            self.params = {'data_dim': 8, 'latent_dim': 2}
    m = Model({'eps_dim':2}, Tester())
    x = tf.constant([[0.0,1,2,3,4,5,6,7]])
    e = tf.constant([[0.0,0]])
    y = tf.constant([[0.0,1]])
    print(m.decoder_network()([y,e]))
    print(m.encoder_network()(x))
    print(m.critic_network()([x,y]))
