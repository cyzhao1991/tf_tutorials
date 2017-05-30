import math
import numpy as np
import gym
import logging
from gym import spaces
from gym.utils import seeding

class ReachingEnv():

	def __init__(self, include_t = False):

		self.gravity = 9.8
		self.length = 0.5 # actually half the pole's length
		self.max_force = 10
		self.dt = 0.02  # seconds between state updates
		self.max_obs = 5.
		self.action_space = spaces.Box(low = -self.max_force, high = self.max_force, shape = (2,))
		self.goal_state = np.array([2., 2., 0., 0.])
		self.seed()
		self.t = 0.
		self.include_t = include_t
		if self.include_t:
			self.observation_space = spaces.Box(-self.max_obs, self.max_obs, shape = (5,))
		else:
			self.observation_space = spaces.Box(-self.max_obs, self.max_obs, shape = (4,))


	def seed(self, seed = None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self,action):
		x,y,xdot,ydot = self.state
		xddot, yddot = action
		newx = x + self.dt*xdot
		newy = y + self.dt*ydot
		newxdot = xdot + self.dt*xddot
		newydot = ydot + self.dt*yddot

		self.state = np.array([newx,newy,newxdot,newydot])
		diff = np.abs(self.state[0:4] - self.goal_state[0:4])
		done = bool((diff<0.05).all())
		reward = - np.linalg.norm(diff) ** 2/16 if np.linalg.norm(diff) < 4 else -1
		self.t = self.t + self.dt

		return self.get_obs(), reward, done, {}


	def reset(self):

		self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,))
		self.steps_beyond_done = None
		self.t = 0.
		return self.get_obs()

	def get_obs(self):
		if self.include_t:
			return np.append(np.array(self.state), self.t)
		else:
			return np.array(self.state)

