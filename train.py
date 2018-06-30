from network import *
from module import *

class Graph:
    def __init__(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.mixture = tf.placeholder(tf.float32, [None, hp.frequency, hp.timestep, hp.num_channel], name='mixture')
            self.true_vocal = tf.placeholder(tf.float32, [None, hp.frequency, hp.timestep, hp.num_channel], name='true_vocal')

            gen, fake_D, real_D = network(self.mixture, self.true_vocal)

            self.wgan_loss = -tf.reduce_mean(real_D) + tf.reduce_mean(fake_D)
            self.gen_loss = -tf.reduce_mean(fake_D)
            self.l1_loss = tf.reduce_mean(tf.abs(self.true_vocal - gen))

            disc_op = tf.train.AdamOptimizer(learning_rate=0.0001)
            gen_op = tf.train.AdamOptimizer(learning_rate=0.0001)

            gen_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="network/generator")
            disc_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="network/discriminator")

            disc_grad = disc_op.compute_gradients(self.wgan_loss, disc_variables)
            gen_grad = gen_op.compute_gradients(self.gen_loss + self.l1_loss, gen_variables)

            self.update_D = disc_op.apply_gradients(disc_grad)
            self.update_G = gen_op.apply_gradients(gen_grad)
            self.clip_D = [var.assign(tf.clip_by_value(var, -0.01, 0.01))for var in disc_variables]

            tf.summary.scalar("generator_loss", self.gen_loss+self.l1_loss)
            tf.summary.scalar("discriminator_loss", self.wgan_loss)

            tf.summary.image("true_mixture", self.mixture)
            tf.summary.image("true_vocal", self.true_vocal)
            tf.summary.image("generated_vocal", gen)

            self.merged = tf.summary.merge_all()

def main():
    mixture = np.load(hp.mixture_data)
    vocals = np.load(hp.vocal_data)

    num_batch = len(mixture) // hp.batch_size

    g = Graph()
    global_step = 0

    with g.graph.as_default():

        saver = tf.train.Saver()

        with tf.Session() as sess:
            if hp.is_restore:
                saver.restore(sess, tf.train.latest_checkpoint(hp.save_dir))
                print("restore successfully!")
            train_writer = tf.summary.FileWriter(hp.save_dir + '/train',
                                                 sess.graph)
            init = tf.global_variables_initializer()
            sess.run(init)

            for epoch in range(hp.num_epochs):

                mixture, vocals = dataset_shuffling(mixture, vocals)
                for i in range(num_batch):
                    batch_mixture, batch_vocal = get_batch(mixture, vocals, i, hp.batch_size)
                    sess.run([g.update_D, g.clip_D], feed_dict={g.mixture:batch_mixture, g.true_vocal:batch_vocal})
                    sess.run(g.update_G, feed_dict={g.mixture:batch_mixture, g.true_vocal:batch_vocal})

                    global_step += 1
                    if global_step % 100 == 0:
                        disc_loss, gen_loss, l1_loss, summary = sess.run([g.wgan_loss, g.gen_loss, g.l1_loss, g.merged], feed_dict={g.mixture: batch_mixture, g.true_vocal: batch_vocal})
                        print("step %d, disc_loss:%.4f, gen_loss:%.4f, l1_loss:%.4f" %(i,disc_loss, gen_loss, l1_loss))
                        saver.save(sess, hp.save_dir+"/model_%d.ckpt" % (global_step))
                        train_writer.add_summary(summary, global_step)


if __name__ == '__main__':
    main()
