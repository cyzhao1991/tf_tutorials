from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import cma, pdb, sys

if 'cma' in sys.argv[1]:
	if '500' in sys.argv[1]:
		pathstr = 'Data/reaching_cma_rbf_500/'
	elif '50' in sys.argv[1]:
		pathstr = 'Data/reaching_cma_rbf_50/'
	reward = []

	for i in range(50):
	    logger = cma.CMADataLogger(pathstr+str(i))
	    logger.load()
	    reward.append(-np.squeeze(logger.f[:,5]))
	    
	x = logger.f[:,1]

	plt.figure(2)
	means = np.mean(reward, axis = 0)
	errors = np.std(reward,axis = 0)
	plt.errorbar(x, means, errors)
	plt.show()
	
elif 'ddpg' in sys.argv[1]:
	if '500' in sys.argv[1]:
		pathstr = 'Data/reaching_ddpg_rbf_transfer_500/'
	elif '50' in sys.argv[1]:
		pathstr = 'Data/reaching_ddpg_rbf_transfer_50/'
	sum_reward = []
	avg_reward = []
	for i in range(7):
		src_reward = np.load(pathstr+str(i)+'.npz')
		sum_reward.append(src_reward['arr_1'][i])
		avg_reward.append(src_reward['arr_0'][i])
	print(np.shape(avg_reward))
	plt.figure(3)
	plt.plot(np.mean(sum_reward, axis = 0))
	plt.plot(np.mean(avg_reward, axis = 0))
	indexis = np.arange(0,500,10)
	errors = np.std(avg_reward, axis = 0)
	means = np.mean(avg_reward, axis = 0)
	plt.errorbar(indexis, means[indexis], errors[indexis])
	plt.axis([0,500, -200, 0])
	plt.show()

pdb.set_trace()