from nlTools import *
from NonlinearVAR import *
from LinearVAR import *
import numpy as np
import pdb
import matplotlib.pyplot as plt

# Data generation
#  generate randomly an instance of A^(p) matrices
N = 5 #number of nodes, network order
P_true = 3
A_true = scaleCoefsUntilStable(np.random.randn(N, N, P_true))
assert(stabilityScore(A_true) < 0.95)

# randomly generate a collection of nonlinearities
M = 4
np.random.seed(seed=2)
nnl_true = [nnl_atRandom(M) for i in range(N)]

sigma = 1
m_y = VAR_realization(A_true, 100, sigma)
m_y_test = VAR_realization(A_true, 100, sigma)

#m_z = np.log(m_y)
m_z1 = np.zeros(m_y.shape)
m_z1_test = np.zeros(m_y_test.shape)
for i in range(N):
    m_z1[i,:] = nnl_true[i].f(m_y[i,:])
    m_z1_test[i,:] = nnl_true[i].f(m_y_test[i,:])
zu = m_z1.max(1)
zl = m_z1.min(1)
#pdb.set_trace()


# Learning
debug_easyMode = 1 #!! in this mode, no need to adapt zl, zu
P_learn = 3
M_learn = 5
Nepochs = 40 #! 100
Nepochs_linear = Nepochs
if debug_easyMode:
     zu = np.maximum(zu, m_z1_test.max(1))
     zl = np.minimum(zl, m_z1_test.min(1))
zu_desired = zu + 0.01*(zu-zl)
zl_desired = zl - 0.01*(zu-zl)
assert((m_z1_test.max(1) < zu_desired).all())
assert((m_z1_test.min(1) > zl_desired).all())
eta = 0.001
eta_linear = eta
nlv_hat, cost_train, cost_test = learn_nonlinearVAR( m_z1, \
     P_learn, M_learn, Nepochs, zl_desired, zu_desired, eta, m_z1_test)
lv_hat, Lcost_train, Lcost_test = learn_linearVAR( m_z1, \
     P_learn, Nepochs_linear, eta_linear, m_z1_test)

#Plotting
plt.figure(1)
plt.plot( np.arange(Nepochs), cost_train  ,label='Nonlinear. Training error')
plt.plot( np.arange(Nepochs), cost_test   ,label='Nonlinear. Test error')
plt.plot( np.arange(Nepochs_linear), Lcost_train ,label='Linear. Training error')
plt.plot( np.arange(Nepochs_linear), Lcost_test  ,label='Linear. Test error')
plt.grid()
plt.legend()
plt.title('Training and test errors')
plt.xlabel('epoch')
plt.ylabel('loss')

plt.show()