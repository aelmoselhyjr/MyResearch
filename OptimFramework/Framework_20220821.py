#!/usr/bin/env python
# coding: utf-8

# # This is the Tensor Train stuff

# In[ ]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 15:13:52 2022

@author: alielmoselhy
"""

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



# # This is the Newton Optimizer

# In[ ]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 15:16:24 2022

@author: alielmoselhy
"""
import numpy as np
import pandas as pd


# this is the Newton cell

def get_utility(X, GC_Const, GC_Obj, const_val, max_order_obj, 
                max_order_const, upper_factor, lower_factor, 
                const_factor):

    H_obj = get_orthogonal_polynomial(X, max_order_obj)
    H_const = get_orthogonal_polynomial(X, max_order_const)
    obj = evaluate_tt_on_grid(GC_Obj, H_obj)
    est_const = evaluate_tt_on_grid(GC_Const, H_const)
    const_penalty = np.exp( (est_const-const_val) / const_factor )
    # axis 0: random trajectories
    # axis 1: parameters

    # X is random trajectories x parameters
    lower_bound = np.exp( -X / lower_factor ).sum(axis = 1)
    upper_bound = np.exp( (X-1)/ upper_factor ).sum(axis = 1)

    #print("bound_error =", upper_bound+lower_bound)
    #print("const_error =", const_penalty)
    #print("Obj_error", obj)
    utility = 1.0*obj + 1.0*const_penalty + 1e2 * (upper_bound + lower_bound) + 0e1 * np.nansum((X - 0.5)**2., axis=1)

    return utility, obj, const_penalty, upper_bound, lower_bound


def linesearch(X, dx, GC_Const, GC_Obj, const_val, 
               max_order_obj, max_order_const, upper_factor,
               lower_factor, const_factor):

    lamb = np.zeros( (X.shape[0], 1) )

    baseline_util, _, _, _, _ = get_utility(X, GC_Const, GC_Obj, const_val, max_order_obj = max_order_obj, 
                                            max_order_const = max_order_const, upper_factor = upper_factor, 
                                            lower_factor = lower_factor, const_factor = const_factor)
    #print("baseline_util =", baseline_util)
    #remaining_X = X
    #for i in np.arange(10): #np.arange(0.1, 1.1, 0.1)[::-1]:
    for i in np.arange(-2.0, 2.0, 0.15):

        #new_X = (X + 2.**(-i * 1.) * dx).clip(0,1)
        new_X = (X + i * dx).clip(0,1)
        if True: #(new_X < 1.).all() and (new_X > 0.).all():            
            temp_util, _, _, _, _ = get_utility(new_X, GC_Const, GC_Obj, const_val, max_order_obj = max_order_obj, 
                                                max_order_const = max_order_const, upper_factor = upper_factor, 
                                                lower_factor = lower_factor, const_factor = const_factor)
            #lamb += 2.**(-i * 1.) * (temp_util<baseline_util).reshape((-1,1)) * (lamb == 0)
            lamb += i * (temp_util<baseline_util).reshape((-1,1)) * (lamb == 0)
        #print(lamb, temp_util)
    #lamb += 2.**(-10.) * (lamb == 0)
    lamb += 0.01 * (lamb == 0)
    return lamb



def get_obj_deriv(GC, X, max_order_obj):
    num_vars = len(GC)

    H   = get_orthogonal_polynomial(X, max_order_obj)
    dH  = get_dH(X, max_order_obj)        
    d2H = get_d2H(X, max_order_obj)
    #dH = get_dH(H, max_order)
    #d2H = get_d2H(dH, max_order)



    #for i in range(num_vars):
    #    H_modified = H.copy()
    #    H_modified[:,i,:] = dH[:,i,:]
    #    jac[:,i] = evaluate_tt_on_grid(GC, H_modified)
    # Just moved this functionality into the first loop of the hessian computation
    func = evaluate_tt_on_grid(GC, H)

    jac = np.zeros(X.shape)
    hess = np.zeros((H.shape[0], num_vars, num_vars))
    for i in range(num_vars):
        #print(i)

        H_modified = H.copy() * 1.0
        H_modified[:,i,:] = dH[:,i,:]
        jac[:,i] = evaluate_tt_on_grid(GC, H_modified)

        H_modified = H.copy()  * 1.0
        H_modified[:,i,:] = d2H[:,i,:]
        hess[:,i,i] = evaluate_tt_on_grid(GC, H_modified)

        for j in range(i):
            H_modified = H.copy()  * 1.0
            H_modified[:,i,:] = dH[:,i,:]
            H_modified[:,j,:] = dH[:,j,:]
            #print("EVAL")
            hess_vals = evaluate_tt_on_grid(GC, H_modified)
            #print("EVAL DONE")
            hess[:,i,j] = hess_vals
            hess[:,j,i] = hess_vals

    if False:
        for i in range(X.shape[0]):
            dx = np.linalg.solve(hess[i], jac[i])
            for lam in np.arange(0.1, 1, 0.1):
                x = X[i] - lam * dx
                H = get_orthogonal_polynomial(x[None], max_order_obj)
                func_new = evaluate_tt_on_grid(GC, H)
                print(lam.round(2), func.round(2), func_new.round(2))
    #print(jac)
    #print(hess)
    #blBL
    return jac, hess

def get_bounds_deriv(X, upper_factor, lower_factor):

    num_points, num_vars = X.shape

    #upper_bound = np.exp(-X/factor)
    #lower_bound = np.exp( (X-1)/factor )

    lower_bound_jac = - np.exp(-X/lower_factor) / lower_factor
    upper_bound_jac = np.exp( (X-1)/upper_factor ) / upper_factor

    combined_jac = lower_bound_jac + upper_bound_jac

    """
    ABOVE IS GOOD
    """

    lower_bound_hess = np.zeros((num_points,num_vars,num_vars))
    upper_bound_hess = np.zeros((num_points,num_vars,num_vars))
    for i in range(num_points):
        lower_bound_hess[i] = np.diag(np.exp(-X[i]/lower_factor) / lower_factor**2.)
        upper_bound_hess[i] = np.diag(np.exp((X[i]-1.)/upper_factor) / upper_factor**2.)

    #lower_bound_hess[:,:,:] = np.eye(num_vars)[None,:,:]
    #lower_bound_hess = lower_bound_hess * np.exp(-X/lower_factor)[:, :, None] / lower_factor**2

    #upper_bound_hess[:,:,:] = np.eye(num_vars)[None,:,:]
    #upper_bound_hess = upper_bound_hess * np.exp( (X-1)/upper_factor )[:, :, None]/ upper_factor**2

    combined_hess = lower_bound_hess + upper_bound_hess


    return combined_jac, combined_hess


def get_const_deriv(GC, X, const_val, max_order_const, const_factor):

    H = get_orthogonal_polynomial(X, max_order_const)
    #dH = get_dH(H, max_order)
    #d2H = get_d2H(dH, max_order)
    dH = get_dH(X, max_order_const)
    d2H = get_d2H(X, max_order_const)

    num_vars = len(GC)

    fx = evaluate_tt_on_grid(GC, H)
    E = np.exp( (fx-const_val)/ const_factor )
    #print(E, E.max(), E.min())
    #print("Cost Penalty:", E)

    jac_complex = np.zeros(X.shape)
    #for i in range(num_vars):
    #    H_modified = H.copy()
    #    H_modified[:,i,:] = dH[:,i,:]
    #    jac_complex[:,i] = evaluate_tt_on_grid(GC, H_modified)
    # moved functionality into first hessian loop


    # H.shape[0] number of random samples
    hess_complex = np.zeros((H.shape[0], num_vars, num_vars))

    # computes the jacobian and Hessian of the const function        
    for i in range(num_vars):
        H_modified = H.copy()
        H_modified[:,i,:] = dH[:,i,:]
        jac_complex[:,i] = evaluate_tt_on_grid(GC, H_modified)
        #print(i)
        for j in range(i):
            H_modified = H.copy()
            H_modified[:,i,:] = dH[:,i,:]               
            d1 = evaluate_tt_on_grid(GC, H_modified)

            H_modified = H.copy()
            H_modified[:,j,:] = dH[:,j,:]
            d2 = evaluate_tt_on_grid(GC, H_modified)
            H_modified[:,i,:] = dH[:,i,:]
            hess_vals = evaluate_tt_on_grid(GC, H_modified)

            #print("EVAL DONE")
            hess_complex[:,i,j] = hess_vals + d1 * d2 / const_factor
            hess_complex[:,j,i] = hess_vals + d1 * d2 / const_factor



        H_modified = H.copy()
        H_modified[:,i,:] = dH[:,i,:]

        squared_term = evaluate_tt_on_grid(GC, H_modified)**2 / const_factor

        H_modified[:,i,:] = d2H[:,i,:]
        hess_complex[:,i,i] = evaluate_tt_on_grid(GC, H_modified) + squared_term

    jac_complex  *= E.reshape(-1,1)  /const_factor
    hess_complex *= E.reshape(-1,1,1)/const_factor

    return jac_complex, hess_complex

    
def gradient_optimize_tt(GC_Const, GC_Obj, n = 1, const_val = 3e6, max_order_obj = 4, 
                         max_order_const=4, upper_factor = 0.01, lower_factor = 0.01,
                         const_factor = 1e4):

    X = np.random.rand(n, len(GC_Const)) * 0.6 + 0.2
    
    
    counter = 0
    for i in range(200):
        #jac, hess = get_jac(GC_Obj, X, max_order)
        #print("JAC")
        #hess = get_hess(GC_Obj, X, max_order)
        #print("HESS")
        obj_jac, obj_hess = get_obj_deriv(GC_Obj, X, max_order_obj)
        const_jac, const_hess = get_const_deriv(GC_Const, X, const_val, max_order_const = max_order_const, 
                                                        const_factor = const_factor)
        boundary_jac, boundary_hess = get_bounds_deriv(X, upper_factor, lower_factor)

        total_jac = 1.0*obj_jac + 1.0*const_jac + 1e2 * boundary_jac + 0e1 * 2. * (X - 0.5)
        total_hess = 1.0*obj_hess + 1.0*const_hess + 1e2 * boundary_hess + 0e1 * 2. * np.eye(X.shape[1])[None]
        dx = np.zeros(X.shape)
        for j in range(X.shape[0]):
            #dx[j] = - np.linalg.solve(np.diag(np.diag(total_hess[j])), total_jac[j])
            dx[j] = - np.linalg.solve(total_hess[j], total_jac[j])
        
        X0 = X.copy()
        lamb = linesearch(X0, dx, GC_Const, GC_Obj, const_val, max_order_obj = max_order_obj, 
                          max_order_const = max_order_const, upper_factor = upper_factor, 
                          lower_factor = lower_factor, const_factor = const_factor)
        
        #H = tt.get_orthogonal_polynomial(X0, max_order)
        print("Lambda:", lamb)
        utility_x0, *_ = get_utility(X0, GC_Const, GC_Obj, const_val = const_val, max_order_obj = max_order_obj, 
                                     max_order_const = max_order_const, upper_factor = upper_factor, 
                                     lower_factor = lower_factor, const_factor = const_factor) 
        print("utility @ x0:", utility_x0)
        #print(evaluate_tt_on_grid(GC_Obj, H))

        X += lamb * dx
        X = X.clip(0,1)
        #print("Params:", X.round(2))

        #H = tt.get_orthogonal_polynomial(X, max_order)
        utility_x, obj, const_penalty, upper_bound, lower_bound = get_utility(X, GC_Const, 
                           GC_Obj, const_val = const_val, max_order_obj = max_order_obj, 
                           max_order_const = max_order_const, upper_factor = upper_factor, 
                           lower_factor = lower_factor, const_factor = const_factor)
        print("utility @ x:", utility_x)
        print("obj @ x:", obj)
        print("const @ x", const_penalty)
        print("upper/lower: ", upper_bound + lower_bound)
        perc_dx = (np.linalg.norm(X-X0)/np.linalg.norm(X0)*100)
        if perc_dx < 0.1:
            counter = counter + 1.
        else:
            counter = 0
        if counter == 5:
            break
        
        print(i, (np.linalg.norm(X-X0)/np.linalg.norm(X0)*100).round(2))
    return X


# # this is the random obtimizer (No Gradient)

# In[ ]:


def sampling_optimize_tt(GC_Const, GC_Obj, const_val, max_order_obj = 4, max_order_const = 4, 
                         num_iters = 1, num_search = 10, xopt = None, to_select = 100):
    Xselected = None
    Y_Obj_opt = 30000
    assert num_iters == 1
    for k in range(num_iters):
        #X = np.random.rand(int(100000/num_iters), len(GC_Const))
        X = np.random.rand(100000, len(GC_Const))
        if xopt is not None:
            M = xopt.shape[0]
            assert X.shape[1] == xopt.shape[1]
            X[:M] = xopt
        for iloop in range(50):  
            # compute objective given set of points
            H_obj = get_orthogonal_polynomial(X, max_order_obj)
            Y_Obj = evaluate_tt_on_grid(GC_Obj, H_obj)
            #print("In Optimization", iloop, Y_Obj.min().round(2),Y_Obj.max().round(2),)
            # compute constraints given set of points
            H_const = get_orthogonal_polynomial(X, max_order_const)
            Y_Const = evaluate_tt_on_grid(GC_Const, H_const)
     
            myCondition = Y_Const < const_val
            S = np.argsort(Y_Obj[myCondition])
            N = len(S)
            Y_Obj_opt = Y_Obj[S[0]]

            S = S[:to_select] # we chose the best performing half of the points
            Xselected = (X[myCondition][S])
            
            for j in range(len(Xselected)):
                X = (0.2 * np.random.rand(to_select,1)) * (np.random.rand(to_select, len(GC_Const))-0.5)
                X[0] = Xselected[j]
                X[1:] = X[1:] + Xselected[j]
                X = X.clip(0,1)

                # compute objective given set of points
                H_obj_temp = get_orthogonal_polynomial(X, max_order_obj)
                Y_Obj_temp = evaluate_tt_on_grid(GC_Obj, H_obj_temp)
                # compute constraints given set of points
                H_const_temp = get_orthogonal_polynomial(X, max_order_const)
                Y_Const_temp = evaluate_tt_on_grid(GC_Const, H_const_temp)
                
                best = np.argsort(Y_Obj_temp[Y_Const_temp < const_val])
                #print(Y_Obj_temp[Y_Const_temp < const_val][best])
                #blabla
                Xselected[j] = X[Y_Const_temp < const_val][best[0]]
            X = Xselected
                
    H_obj = get_orthogonal_polynomial(Xselected, max_order_obj)
    H_const = get_orthogonal_polynomial(Xselected, max_order_const)
    Y_Const = evaluate_tt_on_grid(GC_Const, H_const)
    Y_Obj = evaluate_tt_on_grid(GC_Obj, H_obj)
    S = np.argsort(Y_Obj)

    
    #S = np.argsort(Y_Const < const_val)
    return Xselected[S[:to_select]], Y_Obj[S[:to_select]], Y_Const[S[:to_select]]


# In[46]:


import pandas as pd
import numpy as np
#import model_generation as tt
#import optimizers as optim

def main_loop(const_name, obj_name, const_func, const_val, data_path, rank, max_order_obj, 
              max_order_const, numLoops, upper_factor, lower_factor, const_factor, 
              break_condition, num_iters, num_search, algo_type):
    total_data = pd.read_csv(data_path)
    cols = total_data.columns
    main_loop_counter = 0
    xopt = None
    while True:
        # for the toy example, the below line has no effect on
        print("Main iteration", main_loop_counter)
        #print("Loadings data ...")
        #total_data = total_data.groupby(list(total_data.columns.values[:-2])).mean().reset_index()
        print("Checking data size:", total_data.shape)
        # this part does the fitting
        print("Computing TT models ...")
        GC_Const, tt_rank, Y, Y_estimated, W = main(const_name, total_data, max_order=max_order_const, 
                                                      initial_rank=rank, max_rank=rank, 
                                                      numLoops=numLoops)
        print("Accuracy of the Constraint fit: Y, Y_Est, Diff")
        print(np.linalg.norm(Y).round(2), np.linalg.norm(Y_estimated).round(2), 
              np.linalg.norm(Y-Y_estimated).round(2))
        print("Y_min, Y_max, Y_est_min, Y_est_max")
        print(Y.min(), Y.max(), Y_estimated.min(), Y_estimated.max())
        
        GC_Obj, tt_rank, Y, Y_estimated, W = main(obj_name, total_data, max_order=max_order_obj, 
                                                      initial_rank=rank, max_rank=rank, 
                                                      numLoops=numLoops)
        print("Accuracy of the Obj fit: Y, Y_Est, Diff")
        print(np.linalg.norm(Y).round(2), np.linalg.norm(Y_estimated).round(2), 
              np.linalg.norm(Y-Y_estimated).round(2))
        print("Y_min, Y_max, Y_est_min, Y_est_max")
        print(Y.min(), Y.max(), Y_estimated.min(), Y_estimated.max())
        if algo_type == 'newton':
            xopt = gradient_optimize_tt(GC_Const, GC_Obj, n = 1, const_val = const_val, 
                                        max_order_obj = max_order_obj, max_order_const = max_order_const, upper_factor = upper_factor,
                                        lower_factor = lower_factor, const_factor = const_factor)
            inputs, const, value = const_func(xopt)
            #blah
        elif algo_type == 'stochastic':
            print("Optimizing ...")
            xopt, *_ = sampling_optimize_tt(GC_Const, GC_Obj, const_val, 
                                            max_order_obj = max_order_obj, 
                                            max_order_const = max_order_const, 
                                            num_iters = num_iters, num_search=num_search, 
                                            to_select = 100, xopt = xopt)
            # inputs 1x num_vars: optimal
            # const: scalar value of constraints at optimal
            # value: scalar value of the objective at optimal
            inputs, const, value = const_func(xopt)
            
        print("Optimized!")

        
        print("ACTUAL VALUE:", value.round(2))
        # adds new data point to the dataframe
        #new_frame = pd.DataFrame(data = np.array([*inputs, const, value]).reshape(1,-1), 
        #                         columns = cols)
        # here is the data updated
        print("Updating data ...")
        #total_data = pd.concat([total_data, new_frame])
        cols=[f'x{i}' for i in range(num_vars)]+['obj', 'cost']
        new_frame = pd.DataFrame(np.concatenate([inputs, value[:,None], const[:,None]], axis=1), columns=cols)
        print("Augmenting Frame", new_frame.round(2))
        print("Data Adding!", new_frame.shape)
        total_data = pd.concat([total_data, new_frame], axis=0)
        print("Data Augmenting!", total_data.shape)        
        total_data.to_csv(data_path, index = False)
    
        #print("Paramater Vals:", xopt.round(2))
        print(total_data.sort_values(obj_name)[:5])
        if new_frame[const_name].values[0]>const_val:
            print("Const Violation")
            continue
        elif break_condition(const, value, total_data):
            print("Optimization Done!")
            break
        main_loop_counter = main_loop_counter + 1
    total_data.to_csv("./final_data.csv", index = False)
    total_data.to_csv(data_path, index = False)
    
    print(xopt)
    
    

def toy_func(x):
    print("In Toy function/Showing DIMS", x.shape)
    obj_func = np.nansum((x-0.5)**2, axis=-1)
    constraint_func = np.nansum(x, axis=-1)
    return x, constraint_func, obj_func


def my_break(const, obj, data):
    return False
    if const<1e7 and obj < 0.05:
        return True
    else:
        return False
    
    
def init_toy_problem(num_samples = 10000, num_vars = 30):
    #cols = []
    data_path = "./dummy_func_30.csv"
    #data = pd.DataFrame(data = [], columns = cols)
    #for i in range(30):
    #    cols.append("x"+str(i))
    #cols.append("Const")
    #cols.append("Obj")
    X = np.random.rand(num_samples, num_vars)
    obj = np.nansum((X - 0.5)**2, axis=-1)
    cost = np.nansum(X, axis=-1)
    cols=[f'x{i}' for i in range(num_vars)]+['obj', 'cost']
    pd.DataFrame(np.concatenate((X, obj[:,None], cost[:,None]), axis=1), columns = cols).to_csv("./dummy_func_30.csv", index = False)
    #for i in range(100000):
    #    inputs, const, value = toy_func(np.random.rand(30,))
    #    new_frame = pd.DataFrame(data = np.array([*inputs, const, value]).reshape(1,-1), columns = cols)
    #    data = pd.concat([data, new_frame], ignore_index = True)
    
    
    
    


# In[ ]:


num_samples = 5000
num_vars = 30
init_toy_problem(num_samples, num_vars)
inputs, const, value, total_data = main_loop(const_name="cost", obj_name="obj", const_func=toy_func, 
        const_val=30, data_path="./dummy_func_30.csv", rank=4, max_order_obj=3, 
        max_order_const=4, numLoops=30, upper_factor=0.001, lower_factor=0.001, 
        const_factor=1e4, break_condition=my_break, num_iters = 1, num_search = 10, 
        algo_type = 'newton')


# In[ ]:


get_ipython().run_line_magic('debug', '')


# In[ ]:


cols=[f'x{i}' for i in range(num_vars)]+['obj', 'cost']
pd.concat([total_data, pd.DataFrame(np.concatenate([inputs, const[:,None], value[:,None]], axis=1), columns=cols)], axis=0)


# In[52]:


import pandas as pd
import numpy as np

total_data = pd.read_csv("./dummy_func_30.csv")
test_data = pd.read_csv("./test_30.csv")

for rank in range(1, 11):
    print("RANK:", rank)
    GC_Obj, tt_rank, Y, Y_estimated, W = main("obj", total_data, max_order=3, 
                                                          initial_rank=rank, max_rank=rank, 
                                                          numLoops=50)

    print("Error on TRAINING:", np.linalg.norm(Y-Y_estimated)/np.linalg.norm(Y)*100)

    X = test_data[test_data.columns[:-2].values].values
    Y = test_data["obj"].values
    H = get_orthogonal_polynomial(X, 3)
    Y_estimated = evaluate_tt_on_grid(GC_Obj, H)
    print("Error on Testing:", np.linalg.norm(Y-Y_estimated)/np.linalg.norm(Y)*100)
print("Done!")


# In[53]:


import pandas as pd
import numpy as np

total_data = pd.read_csv("./dummy_func_30.csv")
test_data = pd.read_csv("./test_30.csv")

for rank in range(1, 11):
    print("RANK:", rank)
    GC_Obj, tt_rank, Y, Y_estimated, W = main("cost", total_data, max_order=4, 
                                                          initial_rank=rank, max_rank=rank, 
                                                          numLoops=50)

    print("Error on TRAINING:", np.linalg.norm(Y-Y_estimated)/np.linalg.norm(Y)*100)

    X = test_data[test_data.columns[:-2].values].values
    Y = test_data["cost"].values
    H = get_orthogonal_polynomial(X, 4)
    Y_estimated = evaluate_tt_on_grid(GC_Obj, H)
    print("Error on Testing:", np.linalg.norm(Y-Y_estimated)/np.linalg.norm(Y)*100)
print("Done!")


# In[45]:


num_samples = 5000
num_vars = 30
init_toy_problem(num_samples, num_vars)


# In[51]:


num_samples = 50000
num_vars = 30
data_path = "./test_30.csv"
#data = pd.DataFrame(data = [], columns = cols)
#for i in range(30):
#    cols.append("x"+str(i))
#cols.append("Const")
#cols.append("Obj")
X = np.random.rand(num_samples, num_vars)
obj = np.nansum((X - 0.5)**2, axis=-1)
cost = np.nansum(X, axis=-1)
cols=[f'x{i}' for i in range(num_vars)]+['obj', 'cost']
pd.DataFrame(np.concatenate((X, obj[:,None], cost[:,None]), axis=1), columns = cols).to_csv("./test_30.csv", index = False)


# In[ ]:





# In[ ]:




