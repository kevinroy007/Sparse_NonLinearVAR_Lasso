import numpy as np
from LinearVAR import *
from nlTools import *
from NonlinearVAR import *
import matplotlib.pyplot as plt

def data_generation(N, P_true, M, T= 100, sigma = .2, toBurn = 10000):

     #  randomly generate a collection of A^(p) matrices
     A0 = 0.1*np.random.randn(N, N, P_true)
     for n in range(N):
          A0[n,n]=1
     A_true = scaleCoefsUntilStable(A0)
     assert(stabilityScore(A_true) < 0.95)

     # randomly generate a collection of nonlinearities
     nnl_true = [nnl_atRandom(M) for i in range(N)]

     m_y =      VAR_realization(A_true, T+toBurn, sigma)[:,toBurn:]
     m_y_test = VAR_realization(A_true, T+toBurn, sigma)[:,toBurn:]

     m_z1 = np.zeros(m_y.shape) # Kevin's first nonlinearity: np.log(m_y)
     m_z1_test = np.zeros(m_y_test.shape)
     for i in range(N):
        m_z1[i,:]      = nnl_true[i].f(m_y[i,:])
        m_z1_test[i,:] = nnl_true[i].f(m_y_test[i,:])

     nlv_true = object.__new__(NonlinearVAR)
     nlv_true.A = A_true
     nlv_true.nnl = nnl_true
     return m_z1, m_z1_test, nlv_true

def zu_zl(m_z):
     zu = m_z.max(1)
     zl = m_z.min(1)
     zu_desired = zu + 0.01*(zu-zl)
     zl_desired = zl - 0.01*(zu-zl)   
     return zu_desired, zl_desired

def plot_transfers(axs, nlv, color = 'red'):
     tuple_shape = nlv.A.shape
     N = tuple_shape[0]
     assert N==tuple_shape[1]
     assert axs.shape == (N,N)
     nnl = nlv.nnl
     A = nlv.A[:,:,0]
     assert (len(nnl) == N) and (A.shape == (N, N))
     t = np.arange(-100, 100, 0.1) # x axis
     for n in range(N):
          for n2 in range(N):
               axs[n2, n].plot(nnl[n].f(t), \
                    nnl[n2].f(A[n2, n]*t), color=color)

def trace_scatters(axs, z_in):
     N, N2 = axs.shape; assert N == N2
     N, T = z_in.shape; assert N == N2
     for n in range(N):
          for n2 in range(N):
               axs[n2, n].scatter(\
                    z_in[n, :-1], z_in[n2, 1:], s=.1)

def my_scatters(z_in, nnl=None, A=None):
     N, _ = z_in.shape
     fig, axs = plt.subplots(N, N)
     trace_scatters(axs, z_in)
     if nnl and A is not None:
          plot_transfers(axs, nnl, A)

def compare_transfers(nlv_true, nlv_estim):
     N = len(nlv_true.nnl)
     assert N==len(nlv_estim.nnl)
     fig, axs = plt.subplots(N, N)
     plot_transfers(axs, nlv_true, color = 'blue')
     plot_transfers(axs, nlv_estim)
    
   