from nlTools import sigmoid
import numpy as np
import matplotlib.pyplot as plt
import time
def aa(x):
    return (x/(np.abs(x)+1)+1)/2
    #return 0.5+ x/(2*(np.abs(x)+1))

def aap(x):
    return 1/(2*(np.abs(x)+1)**2)

def sigprime(x):
    return sigmoid(x)*sigmoid(-x)
#

dt = 0.00001 # sampling
t = np.arange(-10, 10, dt) # x axis

# control timings
start_time = time.time()
s1 = aa(t)
time_s1 = time.time() - start_time
print('arithmetic:  ', time_s1, 's')

start_time = time.time()
s1p = aap(t)
time_d1 = time.time() - start_time
print('arith deriv: ', time_d1, 's')

start_time = time.time()
s2 = sigmoid(t)
time_s2 = time.time() - start_time
print('sigmoid:     ', time_s2, 's')

start_time = time.time()
s2p = sigprime(t)
time_d1 = time.time() - start_time
print('sigm deriv:  ', time_d1, 's')

start_time = time.time()
s3 = np.arctan(t)/np.pi + 1/2
time_s3 = time.time() - start_time
print('arctan:      ', time_s3, 's')

start_time = time.time()
s3p = 1/(t**2 + 1)/np.pi
time_d3 = time.time() - start_time
print('atan deriv:  ', time_d3, 's')

# plot functions
fig, axs = plt.subplots(2, 1)

axs[0].plot(t, s1, t, s2, t, s3)
axs[0].set_xlim(-10, 10)
axs[0].set_ylabel('activations')
axs[0].grid(True)

# plot derivatives
axs[1].plot(t, s1p, t, s2p, t, s3p)
axs[1].set_xlim(-10, 10)
axs[1].set_xlabel('x')
axs[1].set_ylabel('derivatives')
axs[1].grid(True)

fig.tight_layout()
plt.show()