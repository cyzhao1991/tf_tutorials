import numpy as np
import random

class ReplayBuffer(object):

	def __init__(self, buffer_size = 1000000, seed = None):
		
		self.buffer_size = buffer_size
		self.seed = seed
		self.buffer = []
		self.count = 0
		if self.seed is not None:
			random.seed(self.seed)

	def add_sample(self, state, action, reward, next_state, time):
		self.buffer.append( (state, action, reward, next_state, time) )
		self.count += 1

	def rand_sample(self, batch_size = 64, seed = None):
		
		if seed is not None:
			self.set_seed(seed)

		if batch_size < self.count:
			sample_batch = random.sample(self.buffer, batch_size)
		else:
			sample_batch = self.buffer

		s_batch = np.array([_[0] for _ in sample_batch])
		a_batch = np.array([_[1] for _ in sample_batch])
		r_batch = np.reshape( np.array([_[2] for _ in sample_batch]), (-1, 1))
		s2_batch = np.array([_[3] for _ in sample_batch])
		t_batch = np.reshape( np.array([_[4] for _ in sample_batch]), (-1, 1))

		return s_batch, a_batch, r_batch, s2_batch, t_batch

	def reset(self):
		self.count = 0
		del self.buffer[:]

	def set_seed(self, seed = None):
		self.seed = seed
		if self.seed is not None:
			random.seed(self.seed)