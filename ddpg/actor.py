import numpy as np
import tensorflow as tf

class ActorNetwork(object):
    
    def __init__(self, sess, state_dim, action_dim, action_bound, hidden_layer_dim = [400, 300], seed = None):
        
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.hidden_layer_dim = hidden_layer_dim
        self.seed = seed 
        self.num_of_layer = len(hidden_layer)
        self.state = tf.placeholder(tf.float32, [None, state_dim], name = 'state')
        
        self.weights = {}
        self.bias = {}
        self.hidden_layer = {-1: self.state}

        for i in range(self.num_of_layer):
        	tf.name_scope('actor_hidden_layer'+str(i))
	        self.weights[i] = tf.Variable(tf.truncated_normal([state_dim, hidden_layer_dim[i]], stddev = 1.0), name = 'weights')
	        self.bias[i] = tf.Variable(tf.truncated_normal([hidden_layer[i]], stddev = 1.0), name = 'bias')
	        self.hidden_layer[i] = tf.nn.relu( tf.matmul( self.hidden_layer[i-1], self.weights[i] ) + self.bias[i] )
    	
    	tf.name_scope('actor_hidden_layer'+str(num_of_layer))
    	self.weights[self.num_of_layer] = tf.Variable(tf.truncated_normal([hidden_layer_dim[-1], action_dim], stddev = 1.0), name = 'weights')
    	self.non_scaled_action = tf.matmul( self.hidden_layer[self.num_of_layer-1], self.weights[self.num_of_layer])
    	self.action = tf.multiply( tf.nn.tanh(self.non_scaled_action), self.action_bound )

    	self.optimizer = tf.train.AdamOptimizer(learning_rate = 0.01, name = 'adam')
    	
    	self.train_op = self.optimizer.apply_gradient()