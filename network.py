from module import *

def network(mixture, true_vocal, scope="network"):
    with tf.variable_scope(scope):
        fake_vocal = generator(mixture)

        fake_D = discriminator(tf.concat([mixture, fake_vocal], axis=-1))
        real_D = discriminator(tf.concat([mixture, true_vocal], axis=-1), reuse=True)

        return fake_vocal, fake_D, real_D
        
def network_prediction(mixture, scope="network"):
    with tf.variable_scope(scope):
        fake_vocal = generator(mixture)
        return fake_vocal
                