import numpy as np
import tensorflow as tf

class RbfActorNetwork(object):

	def __init__(self, sess, state_dim, action_dim, action_bound, hidden_layer_dim = [400, 300], seed = None, tau = 0.001, learning_rate = 0.001):
		
		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_bound = action_bound
		self.hidden_layer_dim = hidden_layer_dim
		self.seed = seed
		self.tau = tau
		self.num_of_layer = len(hidden_layer_dim)


		self.hidden_layer_dim.insert(0, state_dim)

		self.state, self.action = self.build(name = '')
		self.target_state, self.target_action = self.build(name = '_target')


		self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, name = 'actor_adam')
		
		self.actor_paras = [v for v in tf.trainable_variables() if 'actor' in v.name and 'target' not in v.name]
		self.target_actor_paras = [v for v in tf.trainable_variables() if 'actor' in v.name and 'target' in v.name]


		self.critic_gradient = tf.placeholder(tf.float32, [None, self.action_dim], name = 'critic_gradient')
		self.actor_gradient = tf.gradients(self.action, self.actor_paras, grad_ys = -self.critic_gradient,name = 'actor_gradient')

		self.train_op = self.optimizer.apply_gradients(zip(self.actor_gradient, self.actor_paras))

		self.update_actor_paras = [target_param.assign( tf.multiply(target_param,1.-self.tau) + tf.multiply(network_param,self.tau) ) \
			for target_param, network_param in zip(self.target_actor_paras, self.actor_paras) ]


	def build(self, name = None):

		with tf.name_scope('actor'+name):

			state = tf.placeholder(tf.float32, [None, self.state_dim], name = 'state')
			weights_rbf = tf.Variable( tf.truncated_normal([501, 2], stddev = 1.0, seed = self.seed), name = 'weights_rbf' )
			pid_preset = tf.constant([[10, 0],[0,10],[2,0],[0,2]], dtype = tf.float32)
			# weights_pid = tf.Variable( tf.truncated_normal([(self.state_dim - 1) , self.action_dim], stddev = 1.0, seed = self.seed), name = 'weights_pid')
			weights_pid = pid_preset
			rbf_mus = tf.constant(np.array([np.arange(0,5.001,0.01)]), dtype = tf.float32, name = 'mus')
			rbf_sign = tf.constant(-1. ,shape = [1, 501], dtype = tf.float32, name = 'sign')
			rbf_sigma = tf.constant(0.1, dtype = tf.float32, name = 'sigma')
			

			rbf_diff = tf.matmul(tf.expand_dims(state[:,-1],1),rbf_sign) + rbf_mus
			rbf_out = tf.exp( -tf.multiply( tf.square(rbf_diff), .5/tf.square(rbf_sigma)) )
			hidden_input = tf.matmul( rbf_out, weights_rbf )
			desire_pos = tf.multiply( tf.nn.tanh( hidden_input ) , 5.)

			rbf_diff_prev = tf.matmul(tf.expand_dims(state[:,-1],1) - 0.01, rbf_sign) + rbf_mus
			rbf_out_prev = tf.exp( -tf.multiply( tf.square(rbf_diff_prev), .5/tf.square(rbf_sigma)) )
			hidden_input_prev = tf.matmul( rbf_out_prev, weights_rbf )
			desire_pos_prev = tf.multiply( tf.nn.tanh(hidden_input_prev) , 5.)

			rbf_diff_next = tf.matmul(tf.expand_dims(state[:,-1],1) - 0.01, rbf_sign) + rbf_mus
			rbf_out_next = tf.exp( -tf.multiply( tf.square(rbf_diff_next), .5/tf.square(rbf_sigma)) )
			hidden_input_next = tf.matmul( rbf_out_next, weights_rbf )
			desire_pos_next = tf.multiply( tf.nn.tanh(hidden_input_next) , 5.)

			desire_vel = (desire_pos_next - desire_pos_prev) / (2 * 0.01)
			# hidden_input_2 = tf.concat([desire_pos, desire_vel, state[:,:-1]], axis = 1)

			hidden_input_2 = tf.concat([desire_pos, desire_vel],axis = 1) - state[:,:-1]
			 
			non_scaled_action = tf.matmul(hidden_input_2, weights_pid)
			# action  = tf.multiply( tf.nn.tanh(non_scaled_action), self.action_bound )
			action = tf.minimum(tf.maximum(non_scaled_action, -self.action_bound), self.action_bound)

			# weights = {}
			# bias = {}
			# hidden_layer = {-1: state}


			# for i in range(self.num_of_layer):
			# 	weights[i] = tf.Variable(tf.truncated_normal([self.hidden_layer_dim[i], self.hidden_layer_dim[i+1]], stddev = 1.0, seed = self.seed), name = 'weights'+str(i))
			# 	bias[i] = tf.Variable(tf.truncated_normal([self.hidden_layer_dim[i+1]], stddev = 1.0, seed = self.seed), name = 'bias'+str(i))
			# 	hidden_layer[i] = tf.nn.relu( tf.matmul( hidden_layer[i-1], weights[i] ) + bias[i] )

			# weights[self.num_of_layer] = tf.Variable(tf.truncated_normal([self.hidden_layer_dim[-1], self.action_dim], stddev = 0.003, seed = self.seed), name = 'weights'+str(self.num_of_layer))
			# non_scaled_action = tf.matmul( hidden_layer[self.num_of_layer-1], weights[self.num_of_layer] )
			# action = tf.multiply( tf.nn.tanh(non_scaled_action), self.action_bound )

			return state, action


	def train(self, inputs_state, critic_gradient):
		return self.sess.run(self.train_op, feed_dict = {self.state: inputs_state, self.critic_gradient: critic_gradient})


	def predict(self, input_state, if_target = False):
		if if_target:
			return self.sess.run(self.target_action, feed_dict = {self.target_state: input_state})
		else:
			return self.sess.run(self.action, feed_dict = {self.state: input_state})

	def update_target_network(self):
		self.sess.run(self.update_actor_paras)


