import numpy as np
import tensorflow as tf

class ActorCmaLsNetwork(object):
	
	def __init__(self, sess, state_dim, action_dim, action_bound, hidden_layer_dim = [40, 30], seed = None):
		
		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_bound = action_bound
		self.hidden_layer_dim = hidden_layer_dim
		self.seed = seed

		with tf.name_scope('actor_cma'):
			if len(hidden_layer_dim) == 2:

				self.state = tf.placeholder(tf.float32, [None, self.state_dim], name = 'state')
				self.weights1 = tf.placeholder(tf.float32, [self.state_dim, self.hidden_layer_dim[0]], name = 'weights1')
				self.weights2 = tf.placeholder(tf.float32, [self.hidden_layer_dim[0], self.hidden_layer_dim[1]], name = 'weights2')
				self.weights3 = tf.placeholder(tf.float32, [self.hidden_layer_dim[1], self.action_dim], name = 'weights3')
				self.bias1 = tf.placeholder(tf.float32, [self.hidden_layer_dim[0]], name = 'bias1')
				self.bias2 = tf.placeholder(tf.float32, [self.hidden_layer_dim[1]], name = 'bias2')

				self.net1 = tf.nn.relu(tf.matmul(self.state, self.weights1) + self.bias1)
				self.net2 = tf.nn.relu(tf.matmul(self.net1, self.weights2) + self.bias2)
				self.non_scaled_output = tf.matmul(self.net2, self.weights3)
				self.output = tf.multiply(tf.nn.tanh(self.non_scaled_output), self.action_bound)

			elif len(hidden_layer_dim) == 1:

				self.state = tf.placeholder(tf.float32, [None, self.state_dim], name = 'state')
				self.weights1 = tf.placeholder(tf.float32, [self.state_dim, self.hidden_layer_dim[0]], name = 'weights1')
				self.weights2 = tf.placeholder(tf.float32, [self.hidden_layer_dim[0], self.action_dim], name = 'weights2')
				self.bias1 = tf.placeholder(tf.float32, [self.hidden_layer_dim[0]], name = 'bias1')
				# self.bias2 = tf.placeholder(tf.float32, [self.hidden_layer_dim[1]], name = 'bias2')

				self.net1 = tf.nn.relu(tf.matmul(self.state, self.weights1) + self.bias1)
				# self.net2 = tf.nn.relu(tf.matmul(self.net1, self.weights2) + self.bias2)
				self.non_scaled_output = tf.matmul(self.net1, self.weights2)
				self.output = tf.multiply(tf.nn.tanh(self.non_scaled_output), self.action_bound)

			elif len(hidden_layer_dim) == 0:
				self.state = tf.placeholder(tf.float32, [None, self.state_dim], name = 'state')
				self.weights1 = tf.placeholder(tf.float32, [self.state_dim, self.action_dim], name = 'weights1')
				self.bias1 = tf.placeholder(tf.float32, [self.action_dim], name = 'bias1')
				self.non_scaled_output = tf.matmul(self.state, self.weights1) + self.bias1
				self.output = tf.multiply(tf.nn.tanh(self.non_scaled_output), self.action_bound)

	def predict(self, input_state, network):
		if len(self.hidden_layer_dim) == 2:
			feed_dict = {self.state: input_state, self.weights1: network['weights1'], self.weights2:network['weights2'], self.weights3: network['weights3'], self.bias1: network['bias1'], self.bias2: network['bias2']}
		elif len(self.hidden_layer_dim) == 1:
			feed_dict = {self.state: input_state, self.weights1: network['weights1'], self.weights2:network['weights2'], self.bias1: network['bias1']}
		elif len(self.hidden_layer_dim) == 0:
			feed_dict = {self.state: input_state, self.weights1: network['weights1'], self.bias1: network['bias1']}
		return self.sess.run(self.output, feed_dict = feed_dict)


