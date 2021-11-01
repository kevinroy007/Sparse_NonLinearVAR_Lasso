import sys
sys.path.append('code_compare')
import numpy as np
from LinearVAR import scaleCoefsUntilStable
from generating import  nonlinear_VAR_realization
import matplotlib.pyplot as plt
from NonlinearVAR import NonlinearVAR
from LinearVAR_Kevin import learn_model as learn_model_linear
import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph
from learn_model import learn_model
import pdb


p_test = True

if p_test:

    N=5
    M=5
    T=100
    P = 2
    NE = 2
    etanl = 0.0001 
    etal = 0.001
    lamda = 10
    z_data = np.random.rand(N, T)

    A_true =  np.random.rand(N, N, P)

    print(A_true)

    for p in range(P): #sparse initialization of A_true

        g = erdos_renyi_graph(N, 0.5,seed = None, directed= True)  #remove directed after the meeting 
        
        A_t = nx.adjacency_matrix(g)
        A_true[:,:,p] = A_t.todense()
    
#    # pdb.set_trace()
    
    
    z_data =  nonlinear_VAR_realization(A_true, T, np.cbrt, z_data)


    alpha = np.ones((N,M))
    w = np.ones((N,M))
    k = np.ones((N,M))
    b = np.ones((N))
    A = np.ones((N,N,P))

    figure, axis = plt.subplots(2, 2)

    #following command plots without A matrix scaling

    axis[0, 0].plot(z_data[0][:],label = 'sensor 1')
    axis[0, 0].plot(z_data[1][:],label = "sensor 2")
    axis[0, 0].plot(z_data[2][:],label = "sensor 3")
    axis[0, 0].set_title("VAR without A matrix stabilization")
    axis[0, 0].set_xlabel("Time")
    axis[0, 0].set_ylabel("z_data")
    axis[0, 0].legend()
    
    
   
    A_true = scaleCoefsUntilStable(A_true, tol = 0.05, b_verbose = False, inPlace=False)
    z_data =  nonlinear_VAR_realization(A_true, T, np.cbrt, z_data)
     
    #following command plots with A matrix scaling
    
    axis[0, 1].plot(z_data[0][:],label = 'sensor 1')
    axis[0, 1].plot(z_data[1][:],label = "sensor 2")
    axis[0, 1].plot(z_data[2][:],label = "sensor 3")
    axis[0, 1].set_title("VAR with A matrix stabilization")
    axis[0, 1].set_xlabel("Time")
    axis[0, 1].legend()


    #set up the server tomorrow.
    # plt.imshow(A_true[:,:,1])
    # plt.colorbar()
    ##########################################################################################

    newobject = NonlinearVAR(N,M,P,filename_tosave = 'model.nlv') #this line meant for comparing the codes if b_comparing = True. However it runs in both cases.


    cost,A_n= learn_model(NE, etanl ,z_data, A, alpha, w, k, b,lamda, newobject)

    cost_linear,A_l  = learn_model_linear(NE,z_data, A,etal,lamda) 
    
    ##########################################################################################



    t1 = np.arange(NE)+1
    axis[1, 0].plot(t1,cost,label = "NonLinear VAR")
    axis[1, 0].plot(t1,cost_linear,label = "Linear VAR")
    axis[1, 0].set_title("Cost cmparison LinearVAR vs Non LinearVAR")
    axis[1, 0].set_xlabel("Epoch")
    axis[1, 0].set_ylabel("cost")
    axis[1, 0].legend()
    


    t1 = np.arange(NE)+1
    axis[1, 1].plot(t1,cost,label = "NonLinear VAR")
    axis[1, 1].set_title("Cost magnification of Non LinearVAR")
    axis[1, 1].set_xlabel("Epoch")
    axis[1, 1].legend()
    
  
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222) 
    ax3 = fig.add_subplot(223)
    ax1.title.set_text('Ture Adjacency matrix')
    ax2.title.set_text('Adjacency matrix obtained for linear VAR')
    ax3.title.set_text('Adjacency matrix obtained for non linear VAR')
    cb1 =  ax1.imshow(A_true[:,:,1], vmin=0, vmax=10, cmap='jet', aspect='auto')
    cb2 =  ax2.imshow(A_n[:,:,1], vmin=0, vmax=10, cmap='jet', aspect='auto')
    cb3 =  ax3.imshow(A_l[:,:,1], vmin=0, vmax=10, cmap='jet', aspect='auto')
    fig.colorbar(cb1,ax = ax1,orientation='vertical')
    fig.colorbar(cb2,ax = ax2,orientation='vertical')
    fig.colorbar(cb3,ax = ax3, orientation='vertical')
    
    #remove after meeting

    fig = plt.figure()
    nx.draw(g)
    plt.show()
