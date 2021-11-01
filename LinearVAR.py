import numpy as np
import pdb

class LinearVAR:
    def __init__(self, N, P):
            self.A = np.zeros([N, N, P])

    def forward(self, m_z_previous):
            # m_z_previous: {z[t-p]}, p=1..P
            N, P = m_z_previous.shape
            assert(N == self.A.shape[0])
            assert(P == self.A.shape[2])

            m_y_tilde = m_z_previous

            v_y_hat = np.zeros(N)
            for p in range(P):
                v_y_hat = v_y_hat + self.A[:,:, p] @ m_y_tilde[:,p] #!LEq(7)

            v_z_hat = v_y_hat
            
            return v_z_hat

    def compute_cost(self, v_z_hat, v_z_t): 
            v_cost = (v_z_t - v_z_hat)**2
            total_cost = sum(v_cost) #!LEq(9)

            return total_cost

    def backward(self, m_z_previous, v_z_t, v_z_hat):
        N, P = m_z_previous.shape
        assert(N == self.A.shape[0])
        assert(P == self.A.shape[2])
        assert(N == v_z_t.shape[0])

        v_s = 2*(v_z_hat - v_z_t) #LEq(10b)
        dc_dA = np.zeros(self.A.shape)
        for i in range(N):
            for p in range(P):
                dc_dA[i,:,p] = v_s[i]*m_z_previous[:,p] #!LEq(25)
        
        return dc_dA

def learn_linearVAR(m_z_train, P, nEpochs, eta, m_z_test):
    # similar code to that in learn_nonlinearVAR
    
    N, T = m_z_train.shape
    Nt, Tt = m_z_test.shape; assert(Nt == N)
    A_initial = np.zeros([N, N, P])
    
    nlv = LinearVAR(N, P)
    nlv.A = A_initial

    cost_train = np.zeros(nEpochs)
    cost_test  = np.zeros(nEpochs)
    for epoch in range(nEpochs):
        print('Epoch ', epoch)
        cost_thisEpoch_t = np.zeros(T)
        #TEST (forward only)
        for tt in range(P, Tt): #TODO: reduce code repetition
            v_z_tt = m_z_test[:, tt]
            m_z_previous_t = np.zeros([N, P])
            for p in range(P):
                m_z_previous_t[:,p] = m_z_test[:,tt-1-p]
            v_z_hat_t = nlv.forward(m_z_previous_t)   
            #pdb.set_trace()
            cost_thisEpoch_t[tt] = nlv.compute_cost(v_z_hat_t, v_z_tt)/N
        cost_test[epoch] = np.mean(cost_thisEpoch_t)
        
        #TRAIN (forward, backward, and SGD)
        cost_thisEpoch = np.zeros(T)
        for t in range(P, T):           
            v_z_t = m_z_train[:, t]
            m_z_previous = np.zeros([N, P])
            for p in range(P):
                m_z_previous[:,p] = m_z_train[:,t-1-p]
            #
            v_z_hat = nlv.forward(m_z_previous)
            cost_thisEpoch[t] = nlv.compute_cost(v_z_hat, v_z_t)/N
            
            dc_dA = nlv.backward(m_z_previous, v_z_t, v_z_hat)      
            nlv.A = nlv.A - eta*dc_dA
                
        cost_train[epoch] = np.mean(cost_thisEpoch)
    return nlv, cost_train, cost_test

def stabilityScore(m_A):

    assert(m_A.ndim == 3)
    N, N1, P = m_A.shape
    assert(N==N1)
    m_bigA = np.zeros((P*N, P*N))
    m_upperRows = m_A.reshape([N, N*P])
    m_bigA[0:N, :] = m_upperRows
    m_bigA[N:, 0:N*(P-1)] = np.identity(N*(P-1))
    eigenvalues, _ = np.linalg.eig(m_bigA)
    return np.max(np.abs(eigenvalues))

def scaleCoefsUntilStable(t_A_in, tol = 0.05, b_verbose = False, inPlace=False):
    # Takes a 3-way, NxNxP tensor t_A understood as VAR parameter matrices.
    # Iteratively scales it down until it gives rise to a stable VAR process. 
    
    if inPlace:
        t_A = t_A_in
    else:
        t_A = np.copy(t_A_in)
    score = stabilityScore(t_A)
    while score > 1-tol:
        t_A = t_A*(1-tol/1.9)/(score+tol/1.9)
        score = stabilityScore(t_A)
        if b_verbose: print(score)
    return t_A
