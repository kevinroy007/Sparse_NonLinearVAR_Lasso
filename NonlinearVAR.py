from nlTools import *
import numpy as np
import pdb
from projection_simplex import projection_simplex_sort as proj_simplex
import torch
import pickle

class NonlinearVAR:
    def __init__(self, N, M, P, filename_tosave = 'model.nlv'):
        self.A = np.zeros([N, N, P])
        self.nnl = [NodalNonlinearity(M) for m in range(M)]
        self.filename_tosave = filename_tosave

    @staticmethod
    def from_pickle(filename):
        infile = open(filename,'rb')
        new_obj = pickle.load(infile)
        assert type(new_obj) is NonlinearVAR
        infile.close()
        return new_obj

    def to_pickle(self, filename=None):
        if filename is None:
            filename = self.filename_tosave
        outfile = open(filename,'wb')
        pickle.dump(self, outfile)
        outfile.close()

    def forward(self, m_z_previous):
        # m_z_previous: {z[t-p]}, p=1..P
        N, P = m_z_previous.shape
        assert(N == self.A.shape[0])
        assert(P == self.A.shape[2])

        m_y_tilde = np.zeros([N, P])
        for n in range(N):
            for p in range(P):
                m_y_tilde[n, p] = self.nnl[n].g(m_z_previous[n,p]) #!LEq(6)
        
        v_y_hat = np.zeros(N)
        for p in range(P):
            v_y_hat = v_y_hat + self.A[:,:, p] @ m_y_tilde[:,p] #!LEq(7)
        v_z_hat = np.zeros(N)
        for n in range(N):
            v_z_hat[n]= self.nnl[n].f(v_y_hat[n]) #!LEq(8)
        
        return v_z_hat, v_y_hat, m_y_tilde


    def compute_cost(self, v_z_hat, v_z_t):   #could be static 
        v_cost = (v_z_t - v_z_hat)**2
        total_cost = sum(v_cost) #!LEq(9)

        return total_cost

    def backward(self, m_z_previous, v_z_t, v_z_hat, v_y_hat, m_y_tilde):    #kevin changed funciton signature 
        N, P = m_z_previous.shape                         # from tuple_in to v_z_hat, v_y_hat, m_y_tilde
        assert(N == self.A.shape[0])
        assert(P == self.A.shape[2])
        assert(N == v_z_t.shape[0])
        #v_z_hat, v_y_hat, m_y_tilde = tuple_in
        v_s = 2*(v_z_hat - v_z_t) #LEq(10b)

        # Gradients with respect to nonlinearity parameters:
        dc_dalpha = N*[[]]
        dc_dk     = N*[[]]
        dc_dw     = N*[[]]
        dc_db     = N*[[]]
        for i in range(N):
            df_dalpha_i, df_dk_i, df_dw_i, df_db_i = self.nnl[i].gradients_f(v_y_hat[i])
            dc_dalpha[i] = v_s[i]* df_dalpha_i
            dc_dk[i]     = v_s[i]* df_dk_i
            dc_dw[i]     = v_s[i]* df_dw_i
            dc_db[i]     = v_s[i]* df_db_i      
            for p in range(P):
                dg_dalpha, dg_dk, dg_dw, dg_db = self.nnl[i].gradients_g(m_z_previous[i, p])
                for n in range(N):
                    my_f_prime_n = self.nnl[n].f_prime(v_y_hat[n])
                    dc_dalpha[i] = dc_dalpha[i] + v_s[n]*my_f_prime_n*self.A[n, i, p]*dg_dalpha
                    dc_dk[i]     = dc_dk[i]     + v_s[n]*my_f_prime_n*self.A[n, i, p]*dg_dk
                    dc_dw[i]     = dc_dw[i]     + v_s[n]*my_f_prime_n*self.A[n, i, p]*dg_dw
                    dc_db[i]     = dc_db[i]     + v_s[n]*my_f_prime_n*self.A[n, i, p]*dg_db #!LEq(16)
            
        # Gradient with respect to A matrices (3-way tensor form):
        dc_dA = np.zeros(self.A.shape)
        for i in range(N):
            my_f_prime_i = self.nnl[i].f_prime(v_y_hat[i])
            for p in range(P):
                dc_dA[i,:,p] = v_s[i]*my_f_prime_i*m_y_tilde[:,p] #!LEq(25)

        return dc_dalpha, dc_dk, dc_dw, dc_db, dc_dA

    @staticmethod
    def learn(m_z_train, P, M, nEpochs, zl_desired, zu_desired, eta, \
        m_z_test, optimizer_handle = torch.optim.Adam, filename_prefix=''):
        return learn_nonlinearVAR(m_z_train, P, M, nEpochs, \
            zl_desired, zu_desired, eta, m_z_test, optimizer_handle, filename_prefix=filename_prefix)

# will not be used directly
def learn_nonlinearVAR(m_z_train, P, M, nEpochs, \
    zl_desired, zu_desired, eta, m_z_test, optimizer_handle = torch.optim.Adam, filename_prefix=''):
    #b_torch_optim = 1  
    N, T = m_z_train.shape
    Nt, Tt = m_z_test.shape; assert(Nt == N)
    A_initial = np.zeros([N, N, P])
    nnl_initial = [nnl_randomInit(M, zl_desired[n], zu_desired[n])\
         for n in range(N)] # TODO: different initializations
    
    nlv = NonlinearVAR(N, M, P, filename_prefix)
    nlv.A = A_initial
    nlv.nnl = nnl_initial

    cost_train = np.zeros(nEpochs)
    cost_test  = np.zeros(nEpochs)
  
    # conforming Pytorch tensors
    ttg = lambda array: torch.tensor(array, requires_grad = True)
    t_A     =  ttg(nlv.A)
    t_alpha = [ttg(nlv.nnl[i].alpha) for i in range(N)]
    t_k     = [ttg(nlv.nnl[i].k)     for i in range(N)]
    t_w     = [ttg(nlv.nnl[i].w)     for i in range(N)]
    t_b     = [ttg(nlv.nnl[i].b)     for i in range(N)]
    my_parameters = [t_A] + t_alpha + t_k + t_w + t_b
    optimizer = optimizer_handle(my_parameters, lr=eta) #TODO: try rmsprop

    for epoch in range(nEpochs):
        print('Epoch ', epoch)
        cost_test[epoch], cost_train[epoch] = \
            run_an_epoch(nlv, m_z_train, m_z_test, optimizer, \
                t_A, t_alpha, t_k, t_w, t_b, zl_desired, zu_desired)
        nlv.to_pickle(nlv.filename_tosave+str(epoch)+'.nlv')
    return nlv, cost_train, cost_test

def run_an_epoch(nlv, m_z_train, m_z_test, optimizer, \
    t_A, t_alpha, t_k, t_w, t_b, zl_desired, zu_desired):
    N, T = m_z_train.shape
    Nt, Tt = m_z_test.shape; assert(Nt == N)
    P = nlv.A.shape[2]
    cost_thisEpoch_t = np.zeros(T)

    #TEST (forward only)
    if m_z_test is not None:
        for tt in range(P, Tt): #TODO: reduce code repetition
            v_z_tt = m_z_test[:, tt]
            m_z_previous_t = np.zeros([N, P])
            for p in range(P):
                m_z_previous_t[:,p] = m_z_test[:,tt-1-p]
            v_z_hat_t = nlv.forward(m_z_previous_t)[0]     
            cost_thisEpoch_t[tt] = nlv.compute_cost(v_z_hat_t, v_z_tt)/N
        cost_test_out = np.mean(cost_thisEpoch_t)

    #TRAIN (forward, backward, and SGD)
    cost_thisEpoch = np.zeros(T)
    for t in range(P, T):           
        v_z_t = m_z_train[:, t]
        m_z_previous = np.zeros([N, P])
        for p in range(P):
            m_z_previous[:,p] = m_z_train[:,t-1-p]
        #
        v_z_hat, v_y_hat, m_y_tilde = nlv.forward(m_z_previous)     
        tuple_bInfo = (v_z_hat, v_y_hat, m_y_tilde)
        cost_thisEpoch[t] = nlv.compute_cost(v_z_hat, v_z_t)/N
        
        
        dc_dalpha, dc_dk, dc_dw, dc_db, dc_dA = \
            nlv.backward(m_z_previous, v_z_t, tuple_bInfo)
        assert (not(np.isnan(dc_dalpha).any()))

        # Gradient step
        
        #if b_torch_optim:
        t_A.grad = torch.tensor(dc_dA)
        for i in range(N):
            t_alpha[i].grad  = torch.tensor(dc_dalpha[i])
            t_k[i].grad      = torch.tensor(dc_dk[i])
            t_w[i].grad      = torch.tensor(dc_dw[i])
            t_b[i].grad      = torch.tensor(dc_db[i])
        optimizer.step()
        nlv.A = t_A.detach().numpy()
        for i in range(N):
            nlv.nnl[i].alpha = t_alpha[i].detach().numpy()
            nlv.nnl[i].k     = t_k[i].detach().numpy()
            nlv.nnl[i].w     = t_w[i].detach().numpy()
            nlv.nnl[i].b     = t_b[i].detach().numpy()

        #Projection
        for i in range(N):
            nlv.nnl[i].b     = zl_desired[i]
            nlv.nnl[i].alpha = proj_simplex(nlv.nnl[i].alpha, \
                zu_desired[i]-zl_desired[i])
            
            if (abs(nlv.nnl[i].zu() - zu_desired[i])>1e-5).any() \
                or(abs(nlv.nnl[i].zl() - zl_desired[i])>1e-5).any():
                print('projection failed!'); pdb.set_trace()       
            nlv.nnl[i].w     = np.maximum(0.01, nlv.nnl[i].w)
                            
    cost_train_out = np.mean(cost_thisEpoch)
    return cost_test_out, cost_train_out

