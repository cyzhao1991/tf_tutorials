import numpy as np
import tensorflow as tf

class CriticNetwork(object):

	def __init__(self, sess, state_dim, action_dim, hidden_layer_dim = [300], seed = None):

		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.hidden_layer_dim = hidden_layer_dim
		self.num_of_layer = len(hidden_layer_dim)
		self.seed = seed

		self.state = tf.placeholder(tf.float32, [None, self.state_dim], 'state')
		self.action = tf.placeholder(tf.float32, [None, self.action_dim], 'action')

		# self.weights = {}
		# self.bias = {}
		# self.hidden_layer = {-1: self.state}

		# for i in range(self.num_of_layer):
		# 	tf.name_scope('critic_hidden_layer' + str(i))
		# 	self.weights[i] = tf.Variable(tf.truncated_normal([state_dim, hidden_layer_dim[i]], stddev = 1.0), name = 'weights')
		# 	self.bias[i] = tf.Variable(tf.truncated_normal([hidden_layer[i]], stddev = 1.0), name = 'bias')
		# 	self.hidden_layer[i] = tf.nn.relu( tf.matmul( self.hidden_layer[i-1], self.weights[i] ) + self.bias[i])

		# tf.name_scope('critic_hidden_layer' + str(num_of_layer))
		# self.weights[self.num_of_layer] = tf.Variable(tf.truncated_normal([hidden_layer_dim[-1], action_dim], stddev), name = 'weights')
		# self.output = tf.matmul( self.hidden_layer[self.num_of_layer-1], self.weights[self.num_of_layer] )

		self.weights_state = tf.Variable( tf.truncated_normal([state_dim, hidden_layer_dim[0]], stddev = 1.0), name = 'weights_state')
		self.weights_action = tf.Variable( tf.truncated_normal([action_dim, hidden_layer_dim[0]], stddev = 1.0), name = 'weights_action')
		self.bias = tf.Variable( tf.truncated_normal([hidden_layer_dim[0]], stddev = 1.0))

		net1 = tf.matmul( self.state, self.weights_state )
		net2 = tf.matmul( self.action, self.weights_action )
		net = tf.nn.relu( net1 + net2 + self.bias )

		self.output = self.