from nlTools import *
from NonlinearVAR import NonlinearVAR
from LinearVAR import *
from utils_exp import * #data generation and plotting tools
import numpy as np
import pdb
import matplotlib.pyplot as plt
                 
# configuration
np.random.seed(seed=2)
N = 7 #number of nodes, network order
P_true = 1 #!3 #process order
M = 4

debug_easyMode = True #!! in this mode, no need to adapt zl, zu
P_learn = 1 #! 3
M_learn = 4
nExperiments = 1 #! 2
Nepochs = 12 #! 4, 10, 40...
Nepochs_linear = Nepochs

eta = 0.001
eta_linear = eta/10

# representing the training data
# m_z, _, nnl_true1, A1 = data_generation(N, P_true, M)
# my_scatters(m_z, nnl_true1, A1)
# fig4, axs = plt.subplots(N)
# fig5, axs5 = plt.subplots(N)
# for n in range(N):
#      axs[n].plot(m_z[n,:])
#      axs5[n].hist(m_z[n,:])
# plt.show()
# pdb.set_trace()

# Learning
cost_train  = nExperiments*[[]]
cost_test   = nExperiments*[[]]
Lcost_train = nExperiments*[[]]
Lcost_test  = nExperiments*[[]]
var_train   = np.zeros(nExperiments)
var_test    = np.zeros(nExperiments)
for expIndex in range(nExperiments):    
     m_z1, m_z1_test, nlv_true = data_generation(N, P_true, M)
     nlv_true.to_pickle('nlv_true_'+str(expIndex))
     if debug_easyMode:
          zu_desired, zl_desired = zu_zl(np.concatenate((m_z1, m_z1_test), axis=1))
     else:
          zu_desired, zl_desired = zu_zl(m_z1)
     nlv_hat, cost_train[expIndex], cost_test[expIndex]  = NonlinearVAR.learn( m_z1, \
          P_learn, M_learn, Nepochs, zl_desired, zu_desired, eta, m_z1_test, filename_prefix = 'hat'+str(expIndex)+'_')
     lv_hat, Lcost_train[expIndex], Lcost_test[expIndex] = learn_linearVAR(    m_z1, \
          P_learn, Nepochs_linear, eta_linear, m_z1_test)
     var_train[expIndex] = np.var(m_z1.flatten())
     var_test [expIndex] = np.var(m_z1_test.flatten())
     
nmse_train = np.mean(cost_train,  0)/np.mean(var_train)
nmse_Ltrain= np.mean(Lcost_train, 0)/np.mean(var_train)
nmse_test  = np.mean(cost_test,   0)/np.mean(var_test)
nmse_Ltest = np.mean(Lcost_test,  0)/np.mean(var_test)

#Plotting error metrics
plt.figure(1)
plt.plot( np.arange(Nepochs), nmse_train  ,label='Nonlinear. Training NMSE')
plt.plot( np.arange(Nepochs), nmse_test   ,label='Nonlinear. Test NMSE')
plt.plot( np.arange(Nepochs_linear), nmse_Ltrain ,label='Linear. Training NMSE')
plt.plot( np.arange(Nepochs_linear), nmse_Ltest  ,label='Linear. Test NMSE')
plt.grid()
plt.legend()
plt.title('Training and test NMSE')
plt.xlabel('epoch')
plt.ylabel('NMSE')

# plotting functions
compare_transfers(nlv_true, nlv_hat)
plt.show()

#saving

