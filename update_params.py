import numpy as np
import pdb
from compute_gradients import compute_gradients as compute_gradients_n
from compute_gradients_Compare import compute_gradients_compare
from projection_simplex import projection_simplex_sort as proj_simplex

def update_params(eta, z_data, A, alpha, w, k, b, t, z_range,lamda, newobject):
    N,N,P = A.shape
    N,M = k.shape
    #for i in range(N):  # this way of formulation is wrong (loop should be inside backward)

    b_comparing = False # for debugging purposes. For normal execution must be false.
    if b_comparing:
        dC_dA, dc_dalpha, dc_dw, dc_dk, dc_db, cost,hat_z_t = compute_gradients_compare( z_data, A, alpha, w, k, b, t,newobject)
    else:
        dC_dA, dc_dalpha, dc_dw, dc_dk, dc_db, cost,hat_z_t = compute_gradients_n( z_data, A, alpha, w, k, b, t)
    
    # projected SGD (stochastic gradient descent (OPTIMIZER)

    alpha = alpha - eta* dc_dalpha 
    w = w - eta* dc_dw
    k = k - eta* dc_dk
    A  = A - eta*dC_dA - lamda*A.sum()/abs(A).sum()


    # b[i]    = b[i] - eta * dc_db_i TODO
    if np.isnan(alpha).any() or np.isinf(alpha).any():
        print('ERR: found inf or nan in alpha')
        pdb.set_trace()

    #PROJECTION
    
    for i in range(N):
        if (alpha[i,:].sum()  !=  z_range[i]): 
            #projection using the code found online
            try:
                alpha[i][:] = proj_simplex(alpha[i][:], z_range[i])
            except Exception:
                print('ERR: exception at proj_simplex')
                pdb.set_trace()
            if abs(np.sum(alpha[i][:])-z_range[i]) > 1e-5:
                print('ERR: projection failed!'); pdb.set_trace()

            #kevins projection will not be used. We can keep the code here for comparison       
            # alpha1 = cp.Variable(M)     
            # cost_i2 = cp.sum_squares(alpha1 - alpha[i,:])
            # obj = cp.Minimize(cost_i2)

            # constr = [sum(alpha1) == z_range[i]]         
            # opt_val = cp.Problem(obj,constr).solve()    
            # alpha_cvxpy =  np.transpose(alpha1.value)

            #if (np.abs(alpha[i][:]-alpha_cvxpy)>1e-5).any():
            #   print('ERR: projections do not coincide!!'); pdb.set_trace()

    
    #pdb.set_trace()
    

    return A, alpha, w, k, b, cost,hat_z_t
