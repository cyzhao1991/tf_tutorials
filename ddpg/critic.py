import numpy as np
import tensorflow as tf

class CriticNetwork(object):

	def __init__(self, sess, state_dim, action_dim, hidden_layer_dim = [300], l2_alpha = 0.01, seed = None, tau = 0.001, learning_rate = 0.001):

		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.hidden_layer_dim = hidden_layer_dim
		self.num_of_layer = len(hidden_layer_dim)
		self.l2_alpha = l2_alpha
		self.seed = seed
		self.tau = tau
		# self.state = tf.placeholder(tf.float32, [None, self.state_dim], 'state')
		# self.action = tf.placeholder(tf.float32, [None, self.action_dim], 'action')

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

		# self.weights_state = tf.Variable( tf.truncated_normal([state_dim, hidden_layer_dim[0]], stddev = 1.0), name = 'weights_state')
		# self.weights_action = tf.Variable( tf.truncated_normal([action_dim, hidden_layer_dim[0]], stddev = 1.0), name = 'weights_action')
		# self.bias = tf.Variable( tf.truncated_normal([hidden_layer_dim[0]], stddev = 1.0))

		# net1 = tf.matmul( self.state, self.weights_state )
		# net2 = tf.matmul( self.action, self.weights_action )
		# net = tf.nn.relu( net1 + net2 + self.bias )

		# self.output = self.

		self.state, self.action, self.q_value = self.build(name = '')
		self.target_state, self.target_action, self.target_q_value = self.build(name = '_target')

		self.critic_paras = [v for v in tf.trainable_variables() if 'critic' in v.name and 'target' not in v.name]
		self.target_critic_paras = [v for v in tf.trainable_variables() if 'critic' in v.name and 'target' in v.name]

		self.training_q = tf.placeholder(tf.float32, [None, 1], 'training_q')

		self.l2_loss = tf.add_n( [tf.nn.l2_loss(v) for v in self.critic_paras if 'weights' in v.name] )
		self.loss = tf.losses.mean_squared_error( self.training_q, self.q_value ) + self.l2_alpha * self.l2_loss
		self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, name = 'critic_adam')
		self.train_op = self.optimizer.minimize(self.loss)

		self.update_critic_paras = [target_param.assign( tf.multiply(target_param,1.-self.tau) + tf.multiply(network_param,self.tau) ) \
			for target_param, network_param in zip(self.target_critic_paras, self.critic_paras) ]

		self.critic_gradient = tf.gradients(self.q_value, self.action, name = 'critic_gradient')

	def build(self, name = None):
		with tf.name_scope('critic'+name):
			state = tf.placeholder(tf.float32, [None, self.state_dim], 'state')
			action = tf.placeholder(tf.float32, [None, self.action_dim], 'action')

			weights_state  = tf.Variable(tf.truncated_normal([self.state_dim,  self.hidden_layer_dim[0]], stddev = 1.0, seed = self.seed), name = 'weights_state')
			weights_action = tf.Variable(tf.truncated_normal([self.action_dim, self.hidden_layer_dim[0]], stddev = 1.0, seed = self.seed), name = 'weights_action')
			weights_hidden = tf.Variable(tf.truncated_normal([self.hidden_layer_dim[0], 1], stddev = 0.003, seed = self.seed), name = 'weights_hidden')

			bias = tf.Variable(tf.truncated_normal([self.hidden_layer_dim[0]], stddev = 1.0, seed = self.seed), name = 'bias')

			hidden_layer = tf.nn.relu( tf.matmul(state, weights_state) + tf.matmul(action, weights_action) + bias, name = 'hidden_layer')

			q_value = tf.matmul(hidden_layer, weights_hidden)

			return state, action, q_value

	def predict(self, input_state, input_action, if_target=False):
		if if_target:
			return self.sess.run(self.target_q_value, feed_dict = {self.target_state: input_state, self.target_action:input_action})
		else:
			return self.sess.run(self.q_value, feed_dict = {self.state: input_state, self.action:input_action})

	def train(self, input_state, input_action, training_q):
		return self.sess.run([self.train_op, self.loss], feed_dict = {self.state: input_state, self.action: input_action, self.training_q: training_q})

	def update_target_network(self):
		self.sess.run(self.update_critic_paras)

	def compute_critic_gradient(self, input_state, input_action):
		return self.sess.run(self.critic_gradient, feed_dict = {self.state: input_state, self.action: input_action})


