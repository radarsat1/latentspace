
__all__ = ['GradientPenalty']

import tensorflow as tf
import tensorflow.keras as tfk

class GradientPenalty(tfk.layers.Layer):
    def __init__(self, params, critic):
        self.variant = params['variant']
        self.weight = [0.0,params['gp_weight']]['gp' in params['variant']]
        self.critic = critic
        super(GradientPenalty, self).__init__()
    def call(self, x1, x2):
        if self.variant=='0gp': targ = 0
        elif self.variant=='1gp': targ = 1
        else: return tf.constant([0.0])
        r = tf.random.uniform([tf.shape(x1[0])[0],1])
        with tf.GradientTape() as tape:
            x = [x1[0]*r + x2[0]*(1-r),
                 x1[1]*r + x2[1]*(1-r)]
            loss = self.critic(x)
        weights = [w for l in self.critic.layers for w in l.weights if len(l.weights)>0]
        grads = tape.gradient(loss, [weights[0]])[0]
        gradsl2 = tf.sqrt(tf.reduce_sum(grads**2, axis=1))
        return tf.reshape(tf.square(targ - tf.reduce_mean(gradsl2)),(1,1))*self.weight
