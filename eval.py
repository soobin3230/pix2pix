from network import *
from module import *
import os
import numpy as np
import librosa
from data import _get_spectrogram, _rawwave_to_spectrogram, _spectrogram_to__rawwave

class Graph:
    def __init__(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.mixture = tf.placeholder(tf.float32, [None, hp.frequency, hp.timestep, hp.num_channel], name='mixture')

            self.generator = network_prediction(self.mixture)

def main():

    hp.is_training = False
    
    g = Graph()
    rawwave_size = hp.duration * hp.sample_rate
    
    mixture = _get_spectrogram()

    data_length = len(mixture)
    num_spectrograms = data_length // rawwave_size
        
    with g.graph.as_default():

      with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(hp.save_dir))
            print("restore successfully!")
            
            outputs = []
            dex=[]
            for part in mixture:
                part = np.expand_dims(part, axis=0)
                output = sess.run(g.generator, feed_dict={g.mixture:part})
                output =np.squeeze(output, axis=0)
                outputs.append(output)
                dex.append(_spectrogram_to__rawwave(output))
            result=np.asarray(outputs)

            np.save('./data/result.npy',result, allow_pickle=False)
            dex = np.vstack(dex).reshape(-1)
            result = np.squeeze(dex)
            librosa.output.write_wav("./data/result_.wav", result, sr=hp.sample_rate)
            
            init = tf.global_variables_initializer()
            sess.run(init)

if __name__ == '__main__':
    main()
