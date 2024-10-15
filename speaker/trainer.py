import torch, sys, os, tqdm, numpy, soundfile, time, pickle, cv2, glob, random, scipy
import torch.nn as nn
from tools import *
from loss import *
from audiomodel import *
from collections import defaultdict, OrderedDict
from torch.cuda.amp import autocast,GradScaler

def init_trainer(args):
	s = trainer(args)
	args.epoch = 1
	if args.initial_model_a != '':
		print("Model %s loaded from previous state!"%(args.initial_model_a))
		s.load_parameters(args.initial_model_a)

	# If no initial model is given but there are saved models in modelfiles_a,
	# it loads the most recent one (based on naming convention)
	# and sets the epoch number accordingly
	elif len(args.modelfiles_a) >= 1:
		print("Model %s loaded from previous state!"%args.modelfiles_a[-1])
		args.epoch = int(os.path.splitext(os.path.basename(args.modelfiles_a[-1]))[0][6:]) + 1
		s.load_parameters(args.modelfiles_a[-1])

	return s

class trainer(nn.Module):
	def __init__(self, args):
		super(trainer, self).__init__()
		self.speaker_encoder = ECAPA_TDNN(model = args.model_a).cuda()  #  transferred to the GPU with .cuda()

		# n_class: Number of speakers (classes).
		# m: Angular margin.
		# s: Scaling factor.
		# c: Dimensionality of the input embeddings.
		self.speaker_loss    = AAMsoftmax(n_class = args.n_class, m = args.margin_a, s = args.scale_a, c = 192).cuda()

		# This initializes the Adam optimizer, which updates the model's parameters during training.
		# weight_decay helps regularize the model by penalizing large weights.
		self.optim           = torch.optim.Adam(self.parameters(), lr = args.lr, weight_decay = 2e-5)

		# The scheduler reduces the learning rate by a factor (gamma) every test_step epochs.
		# Lowering the learning rate helps the model converge more smoothly.
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = args.test_step, gamma = args.lr_decay)

		# This prints the number of parameters in the speaker encoder model,
		# giving a rough idea of the model's complexity in terms of learnable parameters (measured in millions).
		print(" Speech model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1e6))

	def train_network(self, args):

		# This sets the model to training mode, (from nn.Module.train() documentation):
		# ensuring that layers like batch normalization and dropout behave appropriately
		# (e.g., dropout is active in training but disabled during evaluation).
		self.train()

		# Gradient scaling helps with mixed-precision training,
		# a technique where some operations are done in lower precision (float16) to save memory and improve performance
		scaler = GradScaler()

		# This updates the learning rate based on the current epoch using the StepLR scheduler.
		self.scheduler.step(args.epoch - 1)


		index, top1, loss = 0, 0, 0
		# This gets the current learning rate from the optimizer.
		lr = self.optim.param_groups[0]['lr']


		time_start = time.time()

		# The training data is loaded batch by batch from args.trainLoader.
		#
		for num, (speech, labels) in enumerate(args.trainLoader, start = 1):
			self.zero_grad()  # This clears the gradients of all optimized tensors.
			labels = torch.LongTensor(labels).cuda()  # This transfers the labels to the GPU.

			# Forward pass:
			with autocast():  # This enables automatic mixed precision (AMP) for the forward pass.
				a_embedding   = self.speaker_encoder.forward(speech.cuda(), aug = True)	
				aloss, _      = self.speaker_loss.forward(a_embedding, labels)

			# Backward pass:
			# The loss is scaled using GradScaler and then
			# backpropagated to compute gradients for the model's parameters.
			scaler.scale(aloss).backward()
			scaler.step(self.optim)  # This updates the model's parameters using the computed gradients.
			scaler.update()

			index += len(labels)  # This keeps track of the number of samples processed so far.
			loss += aloss.detach().cpu().numpy()


			time_used = time.time() - time_start


			sys.stderr.write(" [%2d] %.2f%% (est %.1f mins) Lr: %5f, Loss: %.5f\r"%\
			(args.epoch, 100 * (num / args.trainLoader.__len__()), time_used * args.trainLoader.__len__() / num / 60, lr, loss/(num)))
			sys.stderr.flush()
		sys.stdout.write("\n")

		# After each epoch, the learning rate and average loss for that epoch are saved to a score file.
		args.score_file.write("%d epoch, LR %f, LOSS %f\n"%(args.epoch, lr, loss/num))
		args.score_file.flush()
		return
		
	def eval_network(self, args):

		# Sets the model to evaluation mode,
		# disabling behaviors like dropout that should only be active during training.
		self.eval()
		scores_a, labels, res = [], [], []
		embeddings = {}


		lines = open(args.eval_trials).read().splitlines()
		for a_data, filenames in tqdm.tqdm(args.evalLoader, total = len(args.evalLoader)):
			with torch.no_grad():
				a_embedding = self.speaker_encoder.forward(a_data[0].cuda())

				for num in range(len(filenames)):
					filename = filenames[num][0]  # gives the filename number
					# torch.unsqueeze(a_embedding[num], dim=0) adds a new dimension
					# to the tensor a_embedding[num] along the specified axis (dim=0).
					# for a "batch" dim that require 2D dim.
					a = torch.unsqueeze(a_embedding[num], dim = 0)
					# normalizes the embeddings to have a unit norm (L2 normalization).
					embeddings[filename] = F.normalize(a, p=2, dim=1)

		# the system is computing similarity scores
		# between two speaker embeddings (a1 and a2) for verification purposes
		for line in tqdm.tqdm(lines):			
			a1 = embeddings[line.split()[1]]
			a2 = embeddings[line.split()[2]]

			# torch.matmul(a1, a2.T): This computes the dot product (cosine similarity) between the two embeddings.
			# Since both embeddings are normalized, the dot product gives a similarity score.
			# torch.mean(...): Takes the mean of the similarity values across dimensions.
			# .detach().cpu().numpy(): Converts the result to a NumPy array after
			# detaching it from the computational graph and moving it to the CPU
			# (since .numpy() can only be called on CPU tensors).
			score_a = torch.mean(torch.matmul(a1, a2.T)).detach().cpu().numpy()
			scores_a.append(score_a)
			labels.append(int(line.split()[0]))

		for score in [scores_a]:
			# The tuneThresholdfromScore function computes the threshold that balances false acceptances and false rejections,
			# and returns the EER (when False Acceptance Rate = False Rejection Rate).
			EER = tuneThresholdfromScore(score, labels, [1, 0.1])[1]

			fnrs, fprs, thresholds = ComputeErrorRates(score, labels)
			minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
			res.extend([EER, minDCF])
		
		print('EER_a %2.4f, min_a %.4f\n'%(res[0], res[1]))
		args.score_file.write("EER_a %2.4f, min_a %.4f\n"%(res[0], res[1]))
		args.score_file.flush()
		return

	# The save_parameters method in this code saves the model's state (parameters) to a file at the specified path
	def save_parameters(self, path):
		model = OrderedDict(list(self.speaker_encoder.state_dict().items()) + list(self.speaker_loss.state_dict().items()))

		# The torch.save() function serializes and saves the model's state dictionary to the specified file path.
		# This allows the model to be reloaded later for inference or continued training.
		torch.save(model, path)
	# load model parameters from a saved file.
	def load_parameters(self, path):

		# This retrieves the current model's state dictionary,
		# which holds all the model's parameters (weights and biases).
		self_state = self.state_dict()

		# This loads the saved model parameters from the specified path using torch.load().
		loaded_state = torch.load(path)

		# Loop through Saved Parameters:
		# This loop goes through the parameters in the loaded_state dictionary
		for name, param in loaded_state.items():

			# Filter out Unwanted Parameters:
			if ('face_encoder.' not in name) and ('face_loss.' not in name):
				if ('speaker_encoder.' not in name) and ('speaker_loss.' not in name):			
					if name == 'weight':
						name = 'speaker_loss.' + name
					else:
						name = 'speaker_encoder.' + name

				# The loaded parameter values are copied into the current model’s state dictionary,
				# ensuring the model gets updated with the saved values.
				self_state[name].copy_(param)