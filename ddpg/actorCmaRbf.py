import numpy as np
import tensorflow as tf

class ActorCmaRbfNetwork(object):
	
	def __init__(self, sess, state_dim, action_dim, action_bound, hidden_layer_dim = [40, 30], seed = None):
		
		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_bound = action_bound
		self.hidden_layer_dim = hidden_layer_dim
		self.seed = seed

		with tf.name_scope('actor_cma'):
			self.state = tf.placeholder(tf.float32, [None, self.state_dim], name = 'state')
			self.weights_rbf = tf.placeholder(tf.float32, [51,2], name = 'weights_rbf')
			self.weights_pid = tf.constant([[10, 0],[0,10],[2,0],[0,2]], dtype = tf.float32)

			self.rbf_mus = tf.constant(np.array([np.arange(0,5.001,0.1)]), dtype = tf.float32, name = 'mus')
			self.rbf_sign = tf.constant(-1. ,shape = [1, 51], dtype = tf.float32, name = 'sign')
			self.rbf_sigma = tf.constant(0.1, dtype = tf.float32, name = 'sigma')

			self.rbf_diff = tf.matmul(tf.expand_dims(self.state[:,-1],1),self.rbf_sign) + self.rbf_mus
			self.rbf_out = tf.exp( -tf.multiply( tf.square(self.rbf_diff), .5/tf.square(self.rbf_sigma)) )
			self.hidden_input = tf.matmul( self.rbf_out, self.weights_rbf )
			self.desire_pos = tf.multiply( tf.nn.tanh( self.hidden_input ) , 5.)

			self.rbf_diff_prev = tf.matmul(tf.expand_dims(self.state[:,-1],1) - 0.01, self.rbf_sign) + self.rbf_mus
			self.rbf_out_prev = tf.exp( -tf.multiply( tf.square(self.rbf_diff_prev), .5/tf.square(self.rbf_sigma)) )
			self.hidden_input_prev = tf.matmul( self.rbf_out_prev, self.weights_rbf )
			self.desire_pos_prev = tf.multiply( tf.nn.tanh(self.hidden_input_prev) , 5.)

			self.rbf_diff_next = tf.matmul(tf.expand_dims(self.state[:,-1],1) - 0.01, self.rbf_sign) + self.rbf_mus
			self.rbf_out_next = tf.exp( -tf.multiply( tf.square(self.rbf_diff_next), .5/tf.square(self.rbf_sigma)) )
			self.hidden_input_next = tf.matmul( self.rbf_out_next, self.weights_rbf )
			self.desire_pos_next = tf.multiply( tf.nn.tanh(self.hidden_input_next) , 5.)

			self.desire_vel = (self.desire_pos_next - self.desire_pos_prev) / (2 * 0.01)
			# hidden_input_2 = tf.concat([desire_pos, desire_vel, state[:,:-1]], axis = 1)

			self.hidden_input_2 = tf.concat([self.desire_pos, self.desire_vel],axis = 1) - self.state[:,:-1]
			 
			self.non_scaled_action = tf.matmul(self.hidden_input_2, self.weights_pid)
			# action  = tf.multiply( tf.nn.tanh(non_scaled_action), self.action_bound )
			self.output = tf.minimum(tf.maximum(self.non_scaled_action, -self.action_bound), self.action_bound)

	def predict(self, input_state, network):
		# if len(self.hidden_layer_dim) == 2:
		# 	feed_dict = {self.state: input_state, self.weights1: network['weights1'], self.weights2:network['weights2'], self.weights3: network['weights3'], self.bias1: network['bias1'], self.bias2: network['bias2']}
		# elif len(self.hidden_layer_dim) == 1:
		# 	feed_dict = {self.state: input_state, self.weights1: network['weights1'], self.weights2:network['weights2'], self.bias1: network['bias1']}
		# elif len(self.hidden_layer_dim) == 0:
		# 	feed_dict = {self.state: input_state, self.weights1: network['weights1'], self.bias1: network['bias1']}

		feed_dict = {self.state: input_state, self.weights_rbf: network['weights1']}
		return self.sess.run(self.output, feed_dict = feed_dict)


