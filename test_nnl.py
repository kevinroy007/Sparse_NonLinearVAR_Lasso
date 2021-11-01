from nlTools import *
from NonlinearVAR import *
from LinearVAR import *
import numpy as np
import pdb
import matplotlib.pyplot as plt

nnl = NodalNonlinearity(3)
nnl.alpha = np.array([1, 2, 3])
nnl.w     = np.array([0.1, 1, 10])
nnl.k     = np.array([1, 2, 3])
print('lower bound:', nnl.zl())
print('upper bound:', nnl.zu())

print('f(3) = ', nnl.f(3))

print('g(5.772449) = ',nnl.g(5.772449))

print(nnl.gradients_f(5))

print(nnl.gradients_g(5))


# data generation
#  generate randomly an instance of A^(p) matrices
N = 2 #number of nodes, network order
P_true = 2
A_true = np.random.randn(N, N, P_true)
print('Stability score of matrix A: ', stabilityScore(A_true))
A_scaled = scaleCoefsUntilStable(A_true)
#pdb.set_trace()

# randomly generate a collection of nonlinearities
M = 4
nnl_true = [nnl_atRandom(M) for i in range(N)]


sigma = 1
m_y = VAR_realization(A_true, 100, sigma)
#m_z = np.log(m_y)
print('y[t] obtained after generation \n',m_y)
m_y_test = VAR_realization(A_true, 100, sigma)

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
P_learn = 2
M_learn = 3
Nepochs = 10
Nepochs_linear = 2*Nepochs
if debug_easyMode:
     zu = np.maximum(zu, m_z1_test.max(1))
     zl = np.minimum(zl, m_z1_test.min(1))
zu_desired = zu + 0.01*(zu-zl)
zl_desired = zl - 0.01*(zu-zl)
assert((m_z1_test.max(1) < zu_desired).all())
assert((m_z1_test.min(1) > zl_desired).all())
eta = 0.003
nlv_hat, cost_train, cost_test = learn_nonlinearVAR( m_z1, \
     P_learn, M_learn, Nepochs, zl_desired, zu_desired, eta, m_z1_test)
lv_hat, Lcost_train, Lcost_test = learn_linearVAR( m_z1, \
     P_learn, Nepochs_linear, eta, m_z1_test)
#TODO: plot error vs time
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

""" 
plt.figure(figsize=(8,6))
plt.imshow(A_true[:,:,0],cmap="inferno")
plt.title("True A, p = 1")
plt.colorbar()

plt.figure(figsize=(8,6))
plt.imshow(nlv_hat.A[:,:,0],cmap="inferno")
plt.title("Estimated A, p = 1")
plt.colorbar() 

"""

plt.show()