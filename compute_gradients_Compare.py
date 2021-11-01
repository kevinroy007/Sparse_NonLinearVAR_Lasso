import numpy as np
import sys
sys.path.append('code_compare')
from  g_bisection import g_bisection as g_b
import pdb


def compute_gradients_compare(z_data, A, alpha, w, k, b, t, newobject):
   
# COMPUTE_GRADIENTS gets gradients for cost at time t 
# w.r.t. parameters at subnetwork i
# INPUT VARIABLES
# t: integer, is the time instant
# i: integer, is the subnetwork index
# z_data N x T array, containing the given z data
    # N: number of sensors
    # T: number of time instants
    N, T = z_data.shape
# A: N x N x P array, containing VAR parameter matrices
    N,N,P = A.shape
# alpha: N x M array, alpha parameters
    # M: number of units in subnetwork
    N2, M = alpha.shape
    assert(N2 == N)
# w: N x M  array, w parameters
# k: N x M array, k parameters
# b: N array, b parameters
   # gradient of g w.r.t theta     #!LEq(17)
   # Function definitions

    def sigmoid(l):
          
         return  1/(1+np.exp(-l)) 

    def dgalpha(x,i):
        
        return -1*(dfalpha(x,i)/ f_prime(x,i))                   
      
    def dgw(x,i):
        
        return -1*(dfw(x,i)/ f_prime(x,i))

    def dgk(x,i):
        
        return -1*(dfk(x,i)/ f_prime(x,i))

    def dgb(x,i):
        
        return -1*(dfb(x,i)/ f_prime(x,i))
        
    
    # definitiion of f_prime  dfalha, dfw, dfk and dfb (being written as optional just to verify)
    def f_prime(x,i): 

        a=0

        for m in range(M):
            a = a + alpha[i][m] * sigmoid(w[i][m]*x-k[i][m]) * (1-sigmoid(w[i][m]*x-k[i][m]))*(w[i][m])

        return a
       

    def dfalpha(x,i): 

        return sigmoid(w[i][:]*x-k[i][:]) 

        
    
    def dfb(x,i): 

        return 1
    
    def dfk(x,i): 

        return alpha[i][:] * sigmoid(w[i][:]*x-k[i][:]) * (1-sigmoid(w[i][:]*x-k[i][:]))*(-1)

        

    def dfw(x,i): 

        
        return alpha[i][:] * sigmoid(w[i][:]*x-k[i][:]) * (1-sigmoid(w[i][:]*x-k[i][:]))*(x)

    
    # core functions

    def f(x,i): 
        a = b[i]
        for m in range(M):
            a = a + alpha[i][m] * sigmoid(w[i][m]*x-k[i][m])
  
        return a


    def g(x,i):
        return g_b(x, i, alpha, w, k, b)
    





# Forward pass (function evaluation)
    # (You can use your functions f, g, etc)
    #!LEq(7) from the paper

    tilde_y_tm = np.zeros((N, P))   #!LEq(7a) from the paper # here i_prime is used instead of i
    for i_prime in range(N):
        for p in range(1,P+1):
            assert t-p >= 0
            z_i_tmp = z_data[i_prime, t-p]
            tilde_y_tm[i_prime, p-1] = g(z_i_tmp,i_prime)

    z_i_tmp_i = z_data[:,t-P:t]

    v_z_hat, v_y_hat, m_y_tilde = newobject.forward(z_i_tmp_i)

    hat_y_t = np.zeros((N)) #!LEq(7b)
    for i_prime in range(N): 
        for p in range(P):
            for j in range(N):
                 hat_y_t[i_prime] =  hat_y_t[i_prime] + A[i_prime][j][p]*tilde_y_tm[j][p]

    hat_y_t2 =   np.zeros((N)) #!LEq(7b)
    
    for p in range(P):  #this equation is just to comprae different looping and matrix systems. of theq 7b
        hat_y_t2 = hat_y_t2 + A[:,:,p]@tilde_y_tm[:,p]

    if not np.linalg.norm(hat_y_t - hat_y_t2) < 1e-5:
        print("error in looping and matrix")
        pdb.set_trace()
    
    hat_z_t = np.zeros((N)) #!LEq(7c)
    for i_prime in range(N):    
        hat_z_t[i_prime] = f(hat_y_t[i_prime],i_prime)       
    
    print("\nforward compare for t is",t)
    print("\n\nabsolute sum of tilde_y_tm - m_y_tilde",np.sum(np.abs(tilde_y_tm - m_y_tilde)))             
    print("absolute sum of hat_y_t - v_y_hat",np.sum(np.abs(hat_y_t - v_y_hat) ))
    print("absolute sum of hat_z_t - v_z_hat",np.sum(np.abs(hat_z_t - v_z_hat) ))
    
    #pdb.set_trace()


    # computing cost #!LEq(7d) and vector S #!LEq(8)
    cost_i = [0]*T
    
    cost = 0
    S = 2*( hat_z_t - z_data[:,t])/hat_z_t
    for i_prime in range(N):                         
        cost = cost +  np.square(S[i_prime]/2)   

    #####################  adding sparsity to the cost
    

    cost_i[t] = cost/(N*T)

    # cost_i_l[t] = newobject.compute_cost(v_z_hat,z_data[:,t])

    # print("\n\ncost comparison at time t is",np.abs(cost_i[t] - cost_i_l[t]))
    

   # Backward pass (backpropagation)
   # (You can use your functions f_prime, etc)
   #!Leq(17) from the paper 
   #!LEq(13) from the paper 
    
   
     

    dc_dalpha_l, dc_dk_l, dc_dw_l, dc_db_l, dc_dA_l = newobject.backward(z_i_tmp_i,z_data[:,t],v_z_hat,v_y_hat,m_y_tilde)
    
    dc_dalpha = np.zeros((N,M))
    dc_dw =  np.zeros((N,M))
    dc_dk = np.zeros((N,M))
    dc_db = np.zeros((N))

    for i in range(N):

       # look at the equations from the paper and change undercore i to general form carefully
        
        #pdb.set_trace()

        for n in range(N):
            dc_dalpha_i_a = 0
            for p in range(1,P+1): 
                dc_dalpha_i_a = dc_dalpha_i_a + A[n][i][p-1]*dgalpha(z_data[i][t-p],i)
            dc_dalpha[i,:] = dc_dalpha[i,:]  + S[n]*f_prime(hat_y_t[n],n)*dc_dalpha_i_a     
        dc_dalpha[i,:] = dc_dalpha[i,:]  + S[i]*dfalpha(hat_y_t[i],i)  
        #dc_dalpha[i,:] = -1*dc_dalpha[i,:]

        
        
        for n in range(N):
            dc_dw_i_a = 0
            for p in range(1,P+1): 
                dc_dw_i_a = dc_dw_i_a + A[n][i][p-1]*dgw(z_data[i][t-p],i)
            dc_dw[i,:] = dc_dw[i,:] + S[n]*f_prime(hat_y_t[n],n)*dc_dw_i_a 
        dc_dw[i,:] = dc_dw[i,:] + S[i]*dfw(hat_y_t[i],i)

        
        
        for n in range(N):
            dc_dk_i_a = 0
            for p in range(1,P+1): 
                dc_dk_i_a = dc_dk_i_a + A[n][i][p-1]*dgk(z_data[i][t-p],i)
            dc_dk[i,:]= dc_dk[i,:]+ S[n]*f_prime(hat_y_t[n],n)*dc_dk_i_a 
        dc_dk[i,:] = dc_dk[i,:]+ S[i]*dfk(hat_y_t[i],i)

        

        for n in range(N):
            dc_db_i_a = 0
            for p in range(1,P+1): 
                dc_dbi_a = dc_db_i_a + A[n][i][p-1]*dgb(z_data[i][t-p],i)
            dc_db[i] = dc_db[i] + S[n]*f_prime(hat_y_t[n],n)*dc_dbi_a 
        dc_db[i]= dc_db[i] +  S[i]*dfb(hat_y_t[i],i)
    

        

        dC_dA = np.zeros((N,N,P))
        for j in range(N):
            for p in range(1,P+1): 
                dC_dA[i][j][p-1] = S[i]*f_prime(hat_y_t[n],i)*tilde_y_tm[j, p-1] 
                 
        pdb.set_trace()
    
# Hint: numpy has functionality to add/multiply whole vectors. 
# Use it, and your code will be shorter, similar to the paper
# and, more importantly, easier to debug =)

#OUTPUT VARIABLES:
# dc_dalpha_i M array, gradient of cost at time t 
#             w.r.t. parameter vector alpha_i
# dc_dw_i M array, gradient of cost time t
#           w.r.t. parameter vector w_i
# dc_dk_i M array, gradient w.r.t. k_i
# dc_db_i scalar, derivative w.r.t. b_i
    print("\ndC_dA comparison",np.sum(np.abs(dC_dA-dc_dA_l)))
    print("dC_dalpha comparison",np.sum(np.abs(dc_dalpha-dc_dalpha_l)))
    print("dc_dw comparsion",np.sum(np.abs(dc_dw - dc_dw_l)))
    print("dc_dk comparison",np.sum(np.abs(dc_dk-dc_dk_l)))
    
    #pdb.set_trace()
    
    return dC_dA, dc_dalpha, dc_dw, dc_dk, dc_db, cost,hat_z_t
