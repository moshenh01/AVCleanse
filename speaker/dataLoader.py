import glob, numpy, os, random, soundfile, torch, cv2, wave
from scipy import signal
import torchvision.transforms as transforms


# args
def init_loader(args):

	# The vars(args) function converts the args object into a dictionary of its attributes,
	# allowing the train_loader to accept the parameters such as train_path, frame_len, and others.
	trainloader = train_loader(**vars(args))

	# This is the dataset returned from train_loader, containing the training data.
	# This is a PyTorch utility that wraps a dataset (in this case, the trainloader) and provides batching, shuffling,
	# and parallel data loading functionality.
	args.trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)

	# this initializes the evaluation dataset using the eval_loader class or function.
	evalLoader = eval_loader(**vars(args))

	# Testing stage:
	# This is the dataset returned from eval_loader, containing the evaluation data.
	# The evaluation data loader uses a batch size of 1, meaning it processes one example at a time during evaluation.
	# This is typical in tasks like verification or testing,
	# where you want to process and evaluate each example independently.
	args.evalLoader = torch.utils.data.DataLoader(evalLoader, batch_size = 1, shuffle = False, num_workers = args.n_cpu, drop_last = False)
	return args



# This class is responsible for loading and augmenting the training data in the form of audio files.
# It processes audio files, adds various augmentations such as noise and reverberation,
# and returns the audio segments with corresponding speaker labels for training.

class train_loader(object):
	def __init__(self, train_list, train_path, musan_path, rir_path, frame_len, **kwargs):
		# train_list: The path to the file that contains the list of training audio files and their associated labels.
		# train_path: The path to the directory containing the actual training audio files.
		# musan_path: The path to the MUSAN dataset, which contains noise files (speech, music, and general noise) used for data augmentation.
		# rir_path: The path to the room impulse response (RIR) files used to simulate reverberation for audio augmentation.

		self.train_path = train_path
		self.frame_len = frame_len * 160 + 240  # frame_len * 10ms * 16kHz + 15ms * 16kHz

		self.noisetypes = ['noise','speech','music']  # noise types available for augmentation

		# For example: 'noise': [0,15] means noise could be added with an SNR between 0 and 15 dB.
		self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}  # mapping noise types to their signal-to-noise ratio (SNR).

		#  A dictionary specifying how many noise files should be mixed for each noise type.
		# For instance, 'speech': [3,8] means that between 3 and 8 speech noise files will be mixed.
		self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}

		self.noiselist = {}
		# This loads all noise files from the MUSAN dataset directory (musan_path).
		augment_files = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
		# The noise files are categorized by their type (e.g., 'noise', 'speech', or 'music') and stored in the self.noiselist dictionary
		for file in augment_files:
			if file.split('/')[-4] not in self.noiselist:
				self.noiselist[file.split('/')[-4]] = []
			self.noiselist[file.split('/')[-4]].append(file)

		# This loads room impulse response (RIR) files, which are used to simulate reverberation for audio augmentation.
		self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
		self.data_list = []
		self.data_label = []

		# This reads the list of training files and labels from the train_list file.
		lines = open(train_list).read().splitlines()
		# A dictionary "dictkeys" is created to map speaker labels (from the list) to unique integer identifiers.
		dictkeys = list(set([x.split()[0] for x in lines]))		
		dictkeys.sort()
		dictkeys = { key : ii for ii, key in enumerate(dictkeys) }

		for index, line in enumerate(lines):
			speaker_label = dictkeys[line.split()[0]]
			file_name     = line.split()[1]
			self.data_label.append(speaker_label)
			self.data_list.append(file_name)

	# This method is used to fetch a training sample (audio segment)
	# and its corresponding label for a given index in the dataset.
	def __getitem__(self, index):
		file = self.data_list[index]
		label = self.data_label[index]
		segments = self.load_wav(file = file)

		# Convert the augmented audio segment into a PyTorch tensor for training.
		segments = torch.FloatTensor(numpy.array(segments))
		return segments, label

	# This method loads an audio file,
	# and randomly applies one of several augmentations (original, reverb, speech noise, music noise, or general noise).
	def load_wav(self, file):

		# Load the audio file using the soundfile library.
		# This function returns two values:
		# - utterance: the audio waveform as a numpy array
		# - sr: the sampling rate of the audio file
		utterance, _ = soundfile.read(os.path.join(self.train_path, 'wav', file))

		# If the audio is shorter than the desired frame length, pad it with repeated content.
		if utterance.shape[0] <= self.frame_len:
			shortage = self.frame_len - utterance.shape[0]
			utterance = numpy.pad(utterance, (0, shortage), 'wrap')

		# Randomly select a starting frame for the audio segment.
		startframe = random.choice(range(0, utterance.shape[0] - (self.frame_len)))
		# Extract the audio segment based on the starting frame and frame length.

		#Expand Dimensions: numpy.expand_dims(..., axis = 0)
		# adds an extra dimension to the array at the specified axis.
		# In this case, axis = 0 adds a new dimension at the beginning,
		# converting the 1D array into a 2D array with shape (1, frame_len).
		segment = numpy.expand_dims(numpy.array(utterance[int(startframe):int(startframe)+self.frame_len]), axis = 0)

		# Randomly select one of five augmentations:
		augtype = random.randint(0,4)
		if augtype == 0:   # Original no augmentation
			segment = segment
		elif augtype == 1:
			segment = self.add_rev(segment, length = self.frame_len)
		elif augtype == 2:
			segment = self.add_noise(segment, 'speech', length = self.frame_len)
		elif augtype == 3: 
			segment = self.add_noise(segment, 'music', length = self.frame_len)
		elif augtype == 4:
			segment = self.add_noise(segment, 'noise', length = self.frame_len)
		return segment[0]

	def __len__(self):
		return len(self.data_list)

	# applies reverberation to an audio segment by convolving it with a room impulse response (RIR) file.
	def add_rev(self, audio, length):
		rir_file    = random.choice(self.rir_files)
		rir, sr     = soundfile.read(rir_file)
		rir         = numpy.expand_dims(rir.astype(numpy.float),0)
		rir         = rir / numpy.sqrt(numpy.sum(rir**2))
		return signal.convolve(audio, rir, mode='full')[:,:length]

	def add_noise(self, audio, noisecat, length):

		# clean_db: The decibel level of the clean audio segment.

		clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4)
		numnoise    = self.numnoise[noisecat] # The number of noise files to mix for the given noise category.
		# Randomly select a few noise files from the corresponding noise category (speech, music, or noise).
		noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
		noises = []

		# For each noise file, the method:
		# Loads the noise, pads or trims it to the correct length.
		# Scales the noise based on the desired SNR and adds it to the original audio.
		for noise in noiselist:
			noiselength = wave.open(noise, 'rb').getnframes()
			if noiselength <= length:
				noiseaudio, _ = soundfile.read(noise)
				noiseaudio = numpy.pad(noiseaudio, (0, length - noiselength), 'wrap')
			else:
				start_frame = numpy.int64(random.random()*(noiselength-length))
				noiseaudio, _ = soundfile.read(noise, start = start_frame, stop = start_frame + length)
			noiseaudio = numpy.stack([noiseaudio],axis=0)
			noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
			noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
			noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
		noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
		return noise + audio


class eval_loader(object):
	def __init__(self, eval_list, eval_path, **kwargs):
		# eval_list: A file containing a list of evaluation samples and their respective lengths.
		# eval_path: The path to the directory containing the evaluation audio files.

		# self.data_list: This will store the filenames of the evaluation audio files.
		# self.data_length: This will store the lengths of each audio file in seconds.
		# self.eval_path: Stores the path to the evaluation dataset for loading audio files later.
		self.data_list, self.data_length = [], []
		self.eval_path = eval_path

		# Reads the eval_list file line by line.
		# Each line contains a filename and the corresponding length of the audio file in seconds.
		lines = open(eval_list).read().splitlines()
		for line in lines:
			data = line.split()
			self.data_list.append(data[-2])  # filename
			self.data_length.append(float(data[-1]))  # length of the audio file

		# The evaluation data is sorted by the length of the audio files
		# (in ascending order) to make batching more efficient.
		inds = numpy.array(self.data_length).argsort()
		# The data_list and data_length are sorted based on the indices obtained from the argsort function.
		self.data_list, self.data_length = numpy.array(self.data_list)[inds], \
										   numpy.array(self.data_length)[inds]

		# Minibatch Creation:
		# creates minibatches based on the length of the audio files.
		self.minibatch = []
		start = 0
		while True:
			frame_length = self.data_length[start]
			minibatch_size = max(1, int(100 // frame_length))
			# The minibatch size is calculated based on the frame length.
			end = min(len(self.data_list), start + minibatch_size)
			self.minibatch.append([self.data_list[start:end], frame_length])
			if end == len(self.data_list):
				break
			start = end

	def __getitem__(self, index):
		data_lists, frame_length = self.minibatch[index]
		filenames, segments = [], []

		for num in range(len(data_lists)):
			file_name = data_lists[num]
			filenames.append(file_name)

			# sr: The sampling rate of the audio file. for example, 16kHz.
			audio, sr = soundfile.read(os.path.join(self.eval_path, 'wav', file_name))
			if len(audio) < int(frame_length * sr):
				shortage    = int(frame_length * sr) - len(audio) + 1
				audio       = numpy.pad(audio, (0, shortage), 'wrap')
			audio = numpy.array(audio[:int(frame_length * sr)])
			segments.append(audio)

		# Converts the segments list (which is a NumPy array of audio data)
		# into a PyTorch tensor for compatibility with machine learning models.
		segments = torch.FloatTensor(numpy.array(segments))
		return segments, filenames

	def __len__(self):
		return len(self.minibatch)