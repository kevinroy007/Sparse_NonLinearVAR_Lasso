import numpy as np
from update_params import update_params
from projection_simplex import projection_simplex_sort as proj_simplex

def learn_model(NE, eta ,z_data, A, alpha, w, k, b,lamda, newobject): #TODO: make A, alpha, w, k, b optional
    
    N, T = z_data.shape
    N2,N3,P = A.shape
    assert N==N2 and N==N3
    # document inputs
    
    #TODO randomly initializing A, alpha, w, k, b if not given

    z_maximum  = np.zeros(N)
    for i in range(N):
        z_maximum[i] = np.max(z_data[i,:])
    
    z_minimum  = np.zeros(N)
    for i in range(N):
        z_minimum[i] = np.min(z_data[i,:])
    
    z_range = z_maximum-z_minimum

    z_upper = z_maximum + 0.01*z_range
    z_lower = z_minimum - 0.01*z_range
        
    z_difference = z_upper - z_lower
    b = z_lower
    
    for i in range(N):
        alpha[i][:] = proj_simplex(alpha[i][:], z_difference[i])
        # newobject.nnl[i].alpha = alpha[i,:]
        # newobject.nnl[i].b = b[i]
    
    cost_history = np.zeros(NE)
    for epoch in range(NE):  
        print("epoch",epoch)
        cost = np.zeros(T)
        #compare_f = np.zeros(T)
        for t in range(P, T):   
            #pdb.set_trace() 
            # hat_z_t = np.zeros(N)    
            # v_z_hat = np.zeros(N) 
            A, alpha, w, k, b, cost[t],hat_z_t = update_params(eta, z_data, A, alpha, w, k, b, t, z_difference,lamda, newobject)
            # pdb.set_trace()
            # v_z_hat = newobject.forward(z_data)
            # compare_f[t] = sum(abs(hat_z_t - v_z_hat))p
        cost_history[epoch] = sum(cost)/(N*T)
    return  cost_history,A
    
