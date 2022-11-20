import numpy as np
from numpy import einsum, sqrt, dot, array, nonzero, nancumsum, nansum
from numpy import exp, dot, linalg, arange, log10, array, sqrt, diff, random
from numpy import zeros,nan
from numpy.random import randn,rand
from numpy.linalg import norm



import scipy.linalg
import scipy.stats
from scipy.optimize import minimize
import time

import scipy


from numpy.linalg.linalg import solve

#import numba
import pandas as pd

def factorial(n):
    if n<0:
        print("This is wrong. Negative factorial")
    if n <= 1:
        return 1
    else:
        y = 1
        for i in range(1,n+1):
            y *= i
        return y*1.0
    
def initialize_analysis(max_order, output_field, dataset, demean=True):
    # Y is the output
    # X are the points of dimension MC, numModes
    # W is ones size of Y
    #df = pd.read_csv('final_data.csv')
    #X = df[['ART', 'PrEP', 'VMMC']].values
    
    
    #df = pd.read_csv('./all_points_42.csv')
    #df = pd.read_csv('./dummy_func.csv')

    
    #print(df.shape)
    #df = df[['PresProb','Child_6w', 'TestUptake', 'Staging', 
    #         'PreART', 'FastART','On_ART', 'KeepART', 'ARTInterrupted',  
    #         'PrEP', 'VMMC', 'Cost', 'DALY']]
    
    # We may want to change the order of the input columns
    #df = df.groupby(['ART','PrEP', 'VMMC']).mean().reset_index()
    # df = df.sample(10000)
    
    #if demean:
    #    df = dataset.groupby(list(dataset.columns[:-2].values)).mean().reset_index()
    #if not demean:    
    #    M = int(len(df)/2)
    #    df = dataset.sample(M)
    df = dataset
    X = df[df.columns[:-2].values].values #- 0.5
    Y = df[output_field].values
    """
    X = df[df.columns[:-1].values].values #- 0.5
    Y = df[df.columns[-1]].values
    """
    #Y = df['Cost'].values
    W = Y**0.0
    #H = get_taylor_polynomial(X, max_order)
    H = get_orthogonal_polynomial(X, max_order)
    return X,Y,H,W



def get_orthogonal_polynomial(P, max_order):
    #print(P.shape)
    MC, numModes = P.shape
    H = np.zeros((MC, numModes, max_order))
    H[:,:,0] = 1
    H[:,:,1] = P
    for n in range(2,max_order):
        H[:,:, n] = P**n
    for n in range(2,max_order):
        # print([n, factorial(n)])
        H[:,:, n] = H[:,:, n]/factorial(n)

    return H

def get_dH(P, max_order):
    MC, numModes = P.shape
    dH = np.zeros((MC, numModes, max_order))
    #print(P.shape)
    dH[:,:,0] = 0
    dH[:,:,1] = 1
    for n in range(2,max_order):
        dH[:,:, n] = n * P**(n-1)
    for n in range(2,max_order):
        # print([n, factorial(n)])
        dH[:,:, n] = dH[:,:, n]/factorial(n)
    return dH

def get_d2H(P, max_order):
    MC, numModes = P.shape
    d2H = np.zeros((MC, numModes, max_order))
    d2H[:,:,0] = 0
    d2H[:,:,1] = 0
    for n in range(2,max_order):
        d2H[:,:, n] = n * (n-1) * P**(n-2)
    for n in range(2,max_order):
        # print([n, factorial(n)])
        d2H[:,:, n] = d2H[:,:, n]/factorial(n)
    return d2H




#def normalize_H(dxH):
#    for n in range(2,dxH.shape[2]):
#        # print([n, factorial(n)])
#        H[:,:, n] = H[:,:, n]/np.sqrt(factorial(n))

    
def fillLeftInterpolation(left_index, max_order, tt_rank, GC, H):
    '''
    Can I make these much faster
    U is MC * left rank
    GC[i] is r1,r2,max_order
    H is MC, numModes, max_order
    '''
    # the below dimensions should be asserted
    MC = H.shape[0]
    U = np.ones((MC, 1))
    for i in range(left_index):
        # G = np.zeros((tt_rank[i]))
        G = einsum('ijk,Nk->Nij', GC[i], H[:,i])
        #print(G.shape)
        U = einsum('Ni,Nij->Nj', U, G)
        #print(U.shape)
        # if you want you can store the G
    #print(U.shape)
    #if left_index>3:
    #    bla
    return U
        

def fillRightInterpolation(right_index, max_order, tt_rank, GC, H, num_tensors):
    '''
    Can I make these much faster
    '''
    # the below dimensions should be asserted
    MC = H.shape[0]
    V = np.ones((1, MC))
    #print(V.shape)
    for i in range(num_tensors-1,right_index,-1):
        # G = np.zeros((tt_rank[i]))
        G = einsum('ijk,Nk->Nij', GC[i], H[:,i])
        #print(G.shape)
        V = einsum('Nij,jN->iN', G, V)
        #print(V.shape)
        # if you want you can store the G
    #print(V.shape)
    #if right_index>3:
    #    bal
    return V

def fillCoreBasis_smart(left_index, right_index, H):
    # the below dimensions should be asserted
    MC, numModes, max_order = H.shape
    B = einsum('Ni,Nj->Nij', H[:, left_index], H[:, right_index])
    return B

def fillCoreBasis_dumb(left_index, right_index, H):
    # the below dimensions should be asserted
    MC, numModes, max_order = H.shape
    B = H[:, left_index]
    return B

def evaluate_tt_on_grid(GC, H):
    '''
    Can I make these much faster
    U is MC * left rank
    GC[i] is r1,r2,max_order
    H is MC, numModes, max_order
    '''
    # the below dimensions should be asserted
    num_tensors = len(GC)
    MC = H.shape[0]
    U = np.ones((MC, 1))
    for i in range(num_tensors):
        # G = np.zeros((tt_rank[i]))
        G = einsum('ijk,Nk->Nij', GC[i], H[:,i]) # .clip(-10,10), I am not sure clipping makes any sense
        U = einsum('Ni,Nij->Nj', U, G)
        #print(U.shape)
    return U[:,0]



def initialize_train(numModes, initial_rank, max_order):
    GC = {}
    tt_rank = {}
    #GC[0] = rand(1, initial_rank, max_order)
    GC[0] = np.zeros((1, initial_rank, max_order))
    GC[0][0, :, :] = 1.
    tt_rank[0] = [1,initial_rank]
    for i in range(1, numModes-1):
        #GC[i] = rand(initial_rank, initial_rank, max_order)
        GC[i] = np.zeros((initial_rank, initial_rank, max_order))
        GC[i][:, :, :] = np.eye(initial_rank)[:,:,None]
        tt_rank[i] = [initial_rank,initial_rank]
    #GC[numModes-1] = rand(initial_rank, 1, max_order)
    GC[numModes-1] = np.zeros((initial_rank, 1, max_order))
    GC[numModes-1][:, 0, :] = 1.
    tt_rank[numModes-1] = [initial_rank,1]
    return GC, tt_rank






def f7_dumb(B, U, V, W, YW, ):
    MC, max_order = B.shape 
    MC, r1 = U.shape
    r2, MC = V.shape
    #print(U,V,B)
    A = B[:, None, None, :] * U[:, :, None, None] * V.T[:, None, :, None]

    A = A.reshape(MC, r1 * r2 * max_order)
    AW = A * sqrt(W)[:,None]
    #ATWTWA = AW.T.dot(AW).reshape(r1*max_order, r3*max_order, r1*max_order, r3*max_order)
    #ATWTWY = AW.T.dot(YW).reshape(r1*max_order, r3*max_order)
 
    #return ATWTWA, ATWTWY
    return AW, YW


def get_A(left_index, right_index, num_tensors, max_order, tt_rank, GC, H, YW, W,):
    U = fillLeftInterpolation(left_index, max_order, tt_rank, GC, H)
    V = fillRightInterpolation(left_index, max_order, tt_rank, GC, H, num_tensors)
    B = fillCoreBasis_dumb(left_index, right_index, H)
    MC, r1 = U.shape
    r2, MC = V.shape
    #print(U.shape)
    #bla
    MC, max_order = B.shape
    AW, YW = f7_dumb(B,U,V, W, YW)
    return AW, YW

 


    
def solve_leastsquares_dumb(
         left_index, right_index, num_tensors, max_order, tt_rank, GC, 
         H, Y, W, max_rank, randomH):
     '''
     # dimensions of U: MC x r_i-1
     # dimensions of V: r_i+1 x MC
     # dimensions of B: MC x max_order x max_order
     # U[l1, :] * sum(X[i,j] * B[l1,i,j]) * V[:, l1] = Y[l1]
     # kron(V[:, l1]^T, U[l1, :])
     I need to make this function much faster. Preferably by using the einsum as above
     '''
     MC = Y.shape[0]
     r1 = GC[left_index].shape[0]
     r2 = GC[left_index].shape[1]
     #r3 = GC[right_index].shape[1]
     max_order = GC[left_index].shape[2]
     
     YW = Y * sqrt(W)
     #print(YW, Y, W)

     #ATWTWA,ATWTWY = get_AY(left_index, right_index, num_tensors, max_order, tt_rank, GC, H, YW, W, )
     #print(H)
     AW,YW = get_A(left_index, right_index, num_tensors, max_order, tt_rank, GC, H, YW, W, )
     #print(YW)
     s = np.zeros((max_rank,))
     if True:
         #print(r1,r3,max_order)
         #print(ATWTWA.shape, ATWTWY.shape)
         #x = np.linalg.solve(ATWTWA.reshape(r1*max_order * r3*max_order, r1*max_order * r3*max_order)
         #                    + 1e-4 * np.linalg.norm(ATWTWA) * np.eye(r1*max_order * r3*max_order), ATWTWY.reshape(r1*max_order * r3*max_order))
         #u,s,v = np.linalg.svd(AW, full_matrices=False)
         #ll = np.nonzero(s/s[0]<1e-12)[0][0]
         #if not ll:
         #    ll = len(s)
         #ll = 50
         #AWa = (u[:,:ll] * s[:ll]) @ v[:ll]
         #print(AW.shape, AWa.shape, ll, s[:3], s[-3:])
         #print(AW.shape, AWa.shape, ll)
         AWTAW = AW.T @ AW
         P = np.diag(AWTAW)
         m = P.max()
         P = 1/P.clip(m*1e-4, m)
         P = np.diag(P)
         x = np.linalg.solve(P @ AWTAW + 1e-4 * np.eye(AWTAW.shape[0]),  P @ (AW.T @ YW))
         #x = np.linalg.solve(P[None] * AWTAW + 1e-4 * np.eye(AWTAW.shape[0]),  P *  (AW.T @ YW))
         
         
         
         #x = np.linalg.lstsq(AW, YW, rcond=1e-6)[0]

         #print(AW.shape)
         #print(AW)
         #print(AW.max(axis = 1))
         #print(P)
         #print(P.shape)
         #print(P @ AW)
         #blah
         #P = np.diag(1/(AW.max(axis = 1)+1e-2))
         #x = np.linalg.lstsq(P @ AW, P @ YW, rcond=1e-6)[0]
         
         #x = my_lstsq3(GC, max_order, left_index, right_index, num_tensors, tt_rank, AW, YW, randomH)
         #print(left_index)
        
         
         
         #xr = np.linalg.lstsq(AW[:,:ll], YW)[0]
         #x = np.zeros((AW.shape[1]))
         #x[:ll] = xr
         #blabla
         #print(np.linalg.norm(x))
         betaL = x.reshape(r1, r2, max_order)
         betaR = None
     endt = time.time()
     return betaL, betaR
 




def do_main_iteration(numLoops, num_tensors, max_order, max_rank, tt_rank, GC, H, Y, W):
    converged = False
    error_norm = np.zeros((numLoops,))
    have_time = True
    GC0 = GC.copy()
    Y_estimated_old = 0 * Y


    for dummy in range(numLoops):
        #print("Loop = %d"%(dummy))
        #print(dummy, (np.linalg.norm(Y-Y_estimated_old)/np.linalg.norm(Y) * 100).round(2))
        #print(tt_rank)
        # forward and backward pass
        als_list = range(num_tensors-1)
        als_list = range(num_tensors)
        if dummy%2 == 1 and False:
            als_list = als_list[::-1]

        MC = 50000
        randomSamples = np.random.rand(MC, len(GC))
        #randomH = get_taylor_polynomial(randomSamples, max_order)
        randomH = get_orthogonal_polynomial(randomSamples, max_order)


        # do one pass
        for left_index in als_list:
            #print("Sweep %d at cart %d"%(dummy, left_index))
            right_index = left_index + 1
            start = time.time()
            # this is the outer loop where I move from cart to cart.
            # U, V, B, 
            betaL, betaR = solve_leastsquares_dumb(
                    left_index, right_index, num_tensors, max_order, tt_rank, GC, 
                    H, Y.copy(), W, max_rank, randomH)
            GC[left_index] = betaL
            #GC[right_index] = betaR
            GC0 = GC.copy()
            end = time.time()
            tt_rank[left_index] = GC[left_index].shape[:2]
            #tt_rank[right_index] = GC[right_index].shape[:2]


        #print("Computing error after update: error, original, estimate, \n ... + weights")
        Y_estimated = evaluate_tt_on_grid(GC, H)
        #print('%f, \t%f, \t%f, \n%f, \t%f, \t%f'%(scipy.linalg.norm(Y-Y_estimated), scipy.linalg.norm(Y), scipy.linalg.norm(Y_estimated),
        #                                                 scipy.linalg.norm((Y-Y_estimated)*sqrt(W)), 
        #                                                 scipy.linalg.norm(Y*sqrt(W)), scipy.linalg.norm(Y_estimated*sqrt(W)),))
        #print('%f, \t%f, \t%f'%(scipy.linalg.norm((Y_estimated_old-Y_estimated)*sqrt(W)), 
        #                                                 scipy.linalg.norm(Y_estimated_old*sqrt(W)), 
        #                                                 scipy.linalg.norm(Y_estimated*sqrt(W)),))
        #print("%s\n%s\n"%(regress(Y, Y_estimated, W),regress(Y, Y_estimated_old, W),))
        Y_estimated_old = Y_estimated
        


        if converged is True:
            break
    # evaluate the right interpolation
    # evaluate the basis of the core
    # U, V, B, get the unknowns
    # print(GC)
    # at this point I have to evaluate the model
    Y_estimated = evaluate_tt_on_grid(GC, H)
    #print("The error is:= ")
    #print(scipy.linalg.norm(Y-Y_estimated))
    #print(scipy.linalg.norm(Y))
    return GC, tt_rank




def main(output_field, dataset, max_order=5, initial_rank=3, max_rank = 3, numLoops=15):
    '''
    This is the code I run on each partition.
    func here gives us Y,H,W,D
    H is probably the polynomial chaos expansion
    
    '''

    numModes = 8
    if max_order is None:
        max_order = 5 #this is the discretization
    if initial_rank is None:
        initial_rank = 1 # this is the initial and the final rank since in this implementation I don't allow the rank to change
    if max_rank is None:
        max_rank = 4
    if numLoops is None:
        numLoops = 30


    # Y are the samples we are going to fit
    # H  are the polynomial chaos expansions at the samples X
    # W is the weight, we can use all ones as a start
    # D is something I don't remember, probably the dates and we don't need to do anything with them
    X,Y,H,W = initialize_analysis(max_order, output_field, dataset, demean=True)
    numModes = H.shape[1]
    max_order = H.shape[2]
    

    GC, tt_rank = initialize_train(numModes, initial_rank, max_order)
    #print(len(GC))
    GC, tt_rank = do_main_iteration(numLoops, numModes, max_order, max_rank, tt_rank, GC, H, Y, W)
    Y_estimated = evaluate_tt_on_grid(GC, H)
    #regr = regress(Y, Y_estimated, W)
    return GC, tt_rank, Y, Y_estimated, W


