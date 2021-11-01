import numpy as np
import pdb

def sigmoid_old(x):
    z = np.exp(-x)
    return 1 / (1 + z)

def sigmoid(x): # stable sigmoid
    return np.where(x >= 0, 
        1 / (1 + np.exp(-x)), 
        np.exp(x) / (1 + np.exp(x)))

def sigprime(x):
    return sigmoid(x)*sigmoid(-x)

def aa(x):
    return (x/(np.abs(x)+1)+1)/2
    #return 0.5+ x/(2*(np.abs(x)+1))

def aap(x):
    return 1/(2*(np.abs(x)+1)**2)



class NodalNonlinearity:
    def __init__(self, M, activation=sigmoid, activation_prime = sigprime):
        self.M = M
        self.alpha = np.zeros(M)
        self.k     = np.zeros(M)
        self.w     = np.zeros(M)
        self.b     = np.zeros(1)
        self.activation = activation
        self.activation_prime = activation_prime

    def zl(self): # lower bound
        
        return self.b
    
    def zu(self): # upper bound
        
        return np.sum(self.alpha) + self.b

    def f(self, y):
        out = self.b
        for m in range(self.M):
            out = out + self.alpha[m]*self.activation(self.w[m]*y - self.k[m])
        return out

    def f_prime(self, y):
        a = 0
        for m in range(self.M):
            a = a + self.alpha[m] * self.activation_prime(self.w[m]*y-self.k[m])      
        return a
    
    def g(self,z): #hybrid alg between bisection and Newton
        vy = 0
        max_niter = 1000
        stepsize = 1
        assert(z < self.zu())
        assert(z > self.zl())      
        yl = -10
        while self.f(yl) > z:
            yl = yl*10
        yu = 10
        while self.f(yu) < z:
            yu = yu*10
        for iter in range(max_niter): 
            vz = self.f(vy) 
            if vz > z:
                yu = vy
            else:
                yl = vy
            slope = self.f_prime(vy)                            
            if slope < np.abs(vz - z)/(yu - yl): #bisection iteration                           
                vy = (yu + yl) / 2
            else: #Newton iteration
                vy = vy - stepsize*(vz-z)/slope
            if abs(vz-z)<1e-6: #stopping criterion
                break
        #pdb.set_trace()
        return vy

    def gradients_f(self, y):
        sigma_prime_val = self.activation_prime(self.w*y - self.k)
        df_dalpha = self.activation( self.w * y - self.k )
        df_dk     = - self.alpha * sigma_prime_val   # elementwise product
        df_dw     = self.alpha * sigma_prime_val * y # elementwise product
        df_db     = np.array([1])
        return df_dalpha, df_dk, df_dw, df_db

    def gradients_g(self, z):
        y = self.g(z)
        df_dalpha, df_dk, df_dw, df_db = self.gradients_f(y)
        my_f_prime = self.f_prime(y)
        dg_dalpha = -1/my_f_prime * df_dalpha
        dg_dk     = -1/my_f_prime * df_dk
        dg_dw     = -1/my_f_prime * df_dw
        dg_db     = -1/my_f_prime * df_db #!LEq(20)
        return dg_dalpha, dg_dk, dg_dw, dg_db

def nnl_atRandom(M):
    out = NodalNonlinearity(M)
    out.alpha = np.abs(np.random.randn(M))
    out.k     = np.random.randn(M)
    out.w     = np.abs(np.random.randn(M))
    out.b     = np.random.randn(1)
    return out

def nnl_randomInit(M, zl, zu):
    out = nnl_atRandom(M)
    out.b = np.array([zl])
    out.alpha = np.ones(M)*(zu-zl)/M
    out.w = np.ones(M)
    return out

def VAR_realization(A,T, sigma): 
    N,N1,P = A.shape; assert(N == N1)
    u = sigma*np.random.randn(N, T)
    y= np.zeros([N,T]) 
    y[:, 0:P] = np.random.randn(N, P)
    for t in range (P, T):
        for i in range(N):
            y[i][t] = 0 
            for k in range(N):
                for p in range(P):
                    y[i][t] = y[i][t] + A[i][k][p]*y[k][t-p] + u[i][t]
    return y

    