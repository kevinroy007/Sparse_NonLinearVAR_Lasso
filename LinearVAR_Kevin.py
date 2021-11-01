import sys
sys.path.append('code_compare')
import numpy as np
#from LinearVAR import scaleCoefsUntilStable
#from generating import  nonlinear_VAR_realization
#import matplotlib.pyplot as plt
import pdb
from compute_gradients_Linear import compute_gradients as compute_gradients_l

# need to modify yet making 
#data generation remain Nonlinear
#investigate the need for updationg parameters for matrix A
#try to work it out for on your work sheet on ipad

def learn_model(NE,z_data, A_l,eta,lamda): #TODO: make A, alpha, w, k, b optional
    
    N, T = z_data.shape
    N2,N3,P = A_l.shape
    assert N==N2 and N==N3
    # document inputs
    

    
    cost_history = np.zeros(NE)

    for epoch in range(NE):  
        cost = np.zeros(T)
       
        for t in range(P, T):   
            
            dC_dA,cost[t] = compute_gradients_l(z_data, A_l, t)

            A_l = A_l - eta*dC_dA - lamda * A_l.sum()/abs(A_l).sum()
            
        cost_history[epoch] = sum(cost)/(N*T)

    return cost_history,A_l




p_test = False #see that p_test is false important while comparing with linear and nonilnear
#it is important to keep p_test false. because both n and nonlinear need same z_data

if p_test:
    N=3
    M=3
    T=10
    P = 3
    NE = 40
    eta = 0.001
    z_data = np.random.rand(N, T)
    A_true =  np.random.rand(N, N, P)
    A_true = scaleCoefsUntilStable(A_true, tol = 0.05, b_verbose = False, inPlace=False)
    #print ('A_true is: ', A_true)

    alpha = np.ones((N,M))
    w = np.ones((N,M))
    k = np.ones((N,M))
    b = np.ones((N))
    #A = np.ones((N,N,P))

    z_data =  nonlinear_VAR_realization(A_true, T, np.cbrt, z_data)
    #pdb.set_trace()
    # plt.plot(z_data[0][:],label = 'sensor 1')
    # plt.plot(z_data[1][:],label = "sensor 2")
    # plt.plot(z_data[2][:],label = "sensor 3")
    # plt.title("VAR with A matrix stabilization")
    # plt.xlabel("Time")
    # plt.ylabel("z_data")
    # plt.legend()
    # plt.show()
    


    cost = learn_model(NE,z_data, A,eta)
    
    t1 = np.arange(NE)+1
    plt.plot(t1,cost)
    plt.show()