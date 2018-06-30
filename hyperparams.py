filter_height = 4
filter_width = 4
disc_filter_height = 15
disc_filter_width = 3
stride = 2

encoder_hp = [64,128,256,512,512,512,512,512]
decoder_hp = [[4,1,512],[8,2,512],[16,4,512],[32,8,512],[64,16,256],[128,32,128],[256,64,64],[512,128,1]]
disc_hp = [64,128,256,512]

num_channel = 1
keep_prob = 0.5

sample_rate = 16200
duration = 2
frequency = 512
timestep = 128
fft_size = 1022
hop_length = int(fft_size/4)
window_size = fft_size

mixture_data = './data/mixture.npy'
vocal_data = './data/vocal.npy'
eval_wav = './data/test.wav'

batch_size = 32
num_epochs = 100
save_dir = './logdir'

is_restore = False
is_training = True