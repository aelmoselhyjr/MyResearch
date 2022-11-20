import numpy as np
import pandas as pd
import surrogate as tt

# this is the Newton cell

def get_utility(X, GC_Constraint, GC_Obj, constraint_val, max_order_obj, 
                max_order_constraint, upper_factor, lower_factor, 
                constraint_factor):

    H_obj = tt.get_orthogonal_polynomial(X, max_order_obj)
    H_constraint = tt.get_orthogonal_polynomial(X, max_order_constraint)
    obj = tt.evaluate_tt_on_grid(GC_Obj, H_obj)
    est_constraint = tt.evaluate_tt_on_grid(GC_Constraint, H_constraint)
    constraint_penalty = np.exp( (est_constraint-constraint_val) / constraint_factor )
    # axis 0: random trajectories
    # axis 1: parameters

    # X is random trajectories x parameters
    lower_bound = np.exp( -X / lower_factor ).sum(axis = 1)
    upper_bound = np.exp( (X-1)/ upper_factor ).sum(axis = 1)

    #print("bound_error =", upper_bound+lower_bound)
    #print("constraint_error =", constraint_penalty)
    #print("Obj_error", obj)
    utility = 1.0*obj + 1.0*constraint_penalty + 1e2 * (upper_bound + lower_bound) + 0e1 * np.nansum((X - 0.5)**2., axis=1)

    return utility, obj, constraint_penalty, upper_bound, lower_bound


def linesearch(X, dx, GC_Constraint, GC_Obj, constraint_val, 
               max_order_obj, max_order_constraint, upper_factor,
               lower_factor, constraint_factor):

    lamb = np.zeros( (X.shape[0], 1) )

    baseline_util, _, _, _, _ = get_utility(X, GC_Constraint, GC_Obj, constraint_val, max_order_obj = max_order_obj, 
                                            max_order_constraint = max_order_constraint, upper_factor = upper_factor, 
                                            lower_factor = lower_factor, constraint_factor = constraint_factor)
    #print("baseline_util =", baseline_util)
    #remaining_X = X
    #for i in np.arange(10): #np.arange(0.1, 1.1, 0.1)[::-1]:
    for i in np.arange(-2.0, 2.0, 0.15):

        #new_X = (X + 2.**(-i * 1.) * dx).clip(0,1)
        new_X = (X + i * dx).clip(0,1)
        if True: #(new_X < 1.).all() and (new_X > 0.).all():            
            temp_util, _, _, _, _ = get_utility(new_X, GC_Constraint, GC_Obj, constraint_val, max_order_obj = max_order_obj, 
                                                max_order_constraint = max_order_constraint, upper_factor = upper_factor, 
                                                lower_factor = lower_factor, constraint_factor = constraint_factor)
            #lamb += 2.**(-i * 1.) * (temp_util<baseline_util).reshape((-1,1)) * (lamb == 0)
            lamb += i * (temp_util<baseline_util).reshape((-1,1)) * (lamb == 0)
        #print(lamb, temp_util)
    #lamb += 2.**(-10.) * (lamb == 0)
    lamb += 0.01 * (lamb == 0)
    return lamb



def get_obj_deriv(GC, X, max_order_obj):
    num_vars = len(GC)

    H   = tt.get_orthogonal_polynomial(X, max_order_obj)
    dH  = tt.get_dH(X, max_order_obj)        
    d2H = tt.get_d2H(X, max_order_obj)
    #dH = get_dH(H, max_order)
    #d2H = get_d2H(dH, max_order)



    #for i in range(num_vars):
    #    H_modified = H.copy()
    #    H_modified[:,i,:] = dH[:,i,:]
    #    jac[:,i] = evaluate_tt_on_grid(GC, H_modified)
    # Just moved this functionality into the first loop of the hessian computation
    func = tt.evaluate_tt_on_grid(GC, H)

    jac = np.zeros(X.shape)
    hess = np.zeros((H.shape[0], num_vars, num_vars))
    for i in range(num_vars):
        #print(i)

        H_modified = H.copy() * 1.0
        H_modified[:,i,:] = dH[:,i,:]
        jac[:,i] = tt.evaluate_tt_on_grid(GC, H_modified)

        H_modified = H.copy()  * 1.0
        H_modified[:,i,:] = d2H[:,i,:]
        hess[:,i,i] = tt.evaluate_tt_on_grid(GC, H_modified)

        for j in range(i):
            H_modified = H.copy()  * 1.0
            H_modified[:,i,:] = dH[:,i,:]
            H_modified[:,j,:] = dH[:,j,:]
            #print("EVAL")
            hess_vals = tt.evaluate_tt_on_grid(GC, H_modified)
            #print("EVAL DONE")
            hess[:,i,j] = hess_vals
            hess[:,j,i] = hess_vals

    if False:
        for i in range(X.shape[0]):
            dx = np.linalg.solve(hess[i], jac[i])
            for lam in np.arange(0.1, 1, 0.1):
                x = X[i] - lam * dx
                H = tt.get_orthogonal_polynomial(x[None], max_order_obj)
                func_new = tt.evaluate_tt_on_grid(GC, H)
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


def get_constraint_deriv(GC, X, constraint_val, max_order_constraint, constraint_factor):

    H = tt.get_orthogonal_polynomial(X, max_order_constraint)
    #dH = get_dH(H, max_order)
    #d2H = get_d2H(dH, max_order)
    dH = tt.get_dH(X, max_order_constraint)
    d2H = tt.get_d2H(X, max_order_constraint)

    num_vars = len(GC)

    fx = tt.evaluate_tt_on_grid(GC, H)
    E = np.exp( (fx-constraint_val)/ constraint_factor )
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

    # computes the jacobian and Hessian of the constraint function        
    for i in range(num_vars):
        H_modified = H.copy()
        H_modified[:,i,:] = dH[:,i,:]
        jac_complex[:,i] = tt.evaluate_tt_on_grid(GC, H_modified)
        #print(i)
        for j in range(i):
            H_modified = H.copy()
            H_modified[:,i,:] = dH[:,i,:]               
            d1 = tt.evaluate_tt_on_grid(GC, H_modified)

            H_modified = H.copy()
            H_modified[:,j,:] = dH[:,j,:]
            d2 = tt.evaluate_tt_on_grid(GC, H_modified)
            H_modified[:,i,:] = dH[:,i,:]
            hess_vals = tt.evaluate_tt_on_grid(GC, H_modified)

            #print("EVAL DONE")
            hess_complex[:,i,j] = hess_vals + d1 * d2 / constraint_factor
            hess_complex[:,j,i] = hess_vals + d1 * d2 / constraint_factor



        H_modified = H.copy()
        H_modified[:,i,:] = dH[:,i,:]

        squared_term = tt.evaluate_tt_on_grid(GC, H_modified)**2 / constraint_factor

        H_modified[:,i,:] = d2H[:,i,:]
        hess_complex[:,i,i] = tt.evaluate_tt_on_grid(GC, H_modified) + squared_term

    jac_complex  *= E.reshape(-1,1)  /constraint_factor
    hess_complex *= E.reshape(-1,1,1)/constraint_factor

    return jac_complex, hess_complex

    
def gradient_optimize_tt(GC_Constraint, GC_Obj, n = 1, constraint_val = 3e6, max_order_obj = 4, 
                         max_order_constraint=4, upper_factor = 0.01, lower_factor = 0.01,
                         constraint_factor = 1e4):

    X = np.random.rand(n, len(GC_Constraint)) * 0.6 + 0.2
    mask = np.ones(n,)
    counter = 0
    for i in range(200):
        #jac, hess = get_jac(GC_Obj, X, max_order)
        #print("JAC")
        #hess = get_hess(GC_Obj, X, max_order)
        #print("HESS")
        obj_jac, obj_hess = get_obj_deriv(GC_Obj, X, max_order_obj)
        constraint_jac, constraint_hess = get_constraint_deriv(GC_Constraint, X, constraint_val, max_order_constraint = max_order_constraint, 
                                                        constraint_factor = constraint_factor)
        boundary_jac, boundary_hess = get_bounds_deriv(X, upper_factor, lower_factor)

        total_jac = 1.0*obj_jac + 1.0*constraint_jac + 1e2 * boundary_jac + 0e1 * 2. * (X - 0.5)
        total_hess = 1.0*obj_hess + 1.0*constraint_hess + 1e2 * boundary_hess + 0e1 * 2. * np.eye(X.shape[1])[None]
        dx = np.zeros(X.shape)
        for j in range(X.shape[0]):
            #dx[j] = - np.linalg.solve(np.diag(np.diag(total_hess[j])), total_jac[j])
            dx[j] = - np.linalg.solve(total_hess[j], total_jac[j])
        
        X0 = X.copy()
        lamb = linesearch(X0, dx, GC_Constraint, GC_Obj, constraint_val, max_order_obj = max_order_obj, 
                          max_order_constraint = max_order_constraint, upper_factor = upper_factor, 
                          lower_factor = lower_factor, constraint_factor = constraint_factor)
        
        #H = tt.get_orthogonal_polynomial(X0, max_order)
        print("Lambda:", lamb)
        utility_x0, *_ = get_utility(X0, GC_Constraint, GC_Obj, constraint_val = constraint_val, max_order_obj = max_order_obj, 
                                     max_order_constraint = max_order_constraint, upper_factor = upper_factor, 
                                     lower_factor = lower_factor, constraint_factor = constraint_factor) 
        print("utility @ x0:", utility_x0)
        #print(evaluate_tt_on_grid(GC_Obj, H))
        delta = lamb * dx * (mask>0)[:, None]
        X += delta
        X = X.clip(0,1)
        #print("Params:", X.round(2))

        #H = tt.get_orthogonal_polynomial(X, max_order)
        utility_x, obj, constraint_penalty, upper_bound, lower_bound = get_utility(X, GC_Constraint, 
                           GC_Obj, constraint_val = constraint_val, max_order_obj = max_order_obj, 
                           max_order_constraint = max_order_constraint, upper_factor = upper_factor, 
                           lower_factor = lower_factor, constraint_factor = constraint_factor)

            
        print("utility @ x:", utility_x)
        print("obj @ x:", obj)
        print("constraint @ x", constraint_penalty)
        print("upper/lower: ", upper_bound + lower_bound)
        print("mask:", mask)
        perc_dx = (np.linalg.norm(X-X0, axis = -1)/np.linalg.norm(X0, axis = -1)*100)
        for j in range(n):
            if perc_dx[j] < 0.1 and mask[j]!=0:
                mask[j] += 1
            elif perc_dx[j] < 0.1 and mask[j]!=0:
                mask[j] = 1
            if mask[j]==6:
                mask[j] = 0
            #mask[j] += (perc_dx[j]<0.01)*1 + (perc_dx[j]>0.01)
        #if perc_dx < 0.1:
        #    counter = counter + 1.
        #else:
        #    counter = 0
        #if counter == 5:
        #    break
        if not np.any(mask):
            break
        
        print(i, perc_dx.round(2))
    return X


# # this is the random obtimizer (No Gradient)


def sampling_optimize_tt(GC_Constraint, GC_Obj, constraint_val, max_order_obj = 4, max_order_constraint = 4, 
                         num_optims = 10, xopt = None, to_select = 100):
    Xselected = None
    #X = np.random.rand(int(100000/num_iters), len(GC_Constraint))
    X = np.random.rand(100000, len(GC_Constraint))
    if xopt is not None:
        M = xopt.shape[0]
        assert X.shape[1] == xopt.shape[1]
        X[:M] = xopt
    for iloop in range(50):  
        # compute objective given set of points
        H_obj = tt.get_orthogonal_polynomial(X, max_order_obj)
        Y_Obj = tt.evaluate_tt_on_grid(GC_Obj, H_obj)
        #print("In Optimization", iloop, Y_Obj.min().round(2),Y_Obj.max().round(2),)
        # compute constraintraints given set of points
        H_constraint = tt.get_orthogonal_polynomial(X, max_order_constraint)
        Y_Constraint = tt.evaluate_tt_on_grid(GC_Constraint, H_constraint)
 
        myCondition = Y_Constraint < constraint_val
        S = np.argsort(Y_Obj[myCondition])
        S = S[:to_select] # we chose the best performing half of the points
        Xselected = (X[myCondition][S])
        
        for j in range(len(Xselected)):
            X = (0.2 * np.random.rand(to_select,1)) * (np.random.rand(to_select, len(GC_Constraint))-0.5)
            X[0] = Xselected[j]
            X[1:] = X[1:] + Xselected[j]
            X = X.clip(0,1)

            # compute objective given set of points
            H_obj_temp = tt.get_orthogonal_polynomial(X, max_order_obj)
            Y_Obj_temp = tt.evaluate_tt_on_grid(GC_Obj, H_obj_temp)
            # compute constraintraints given set of points
            H_constraint_temp = tt.get_orthogonal_polynomial(X, max_order_constraint)
            Y_Constraint_temp = tt.evaluate_tt_on_grid(GC_Constraint, H_constraint_temp)
            
            best = np.argsort(Y_Obj_temp[Y_Constraint_temp < constraint_val])
            #print(Y_Obj_temp[Y_Constraint_temp < constraint_val][best])
            #blabla
            Xselected[j] = X[Y_Constraint_temp < constraint_val][best[0]]
        X = Xselected
            
    H_obj = tt.get_orthogonal_polynomial(Xselected, max_order_obj)
    H_constraint = tt.get_orthogonal_polynomial(Xselected, max_order_constraint)
    Y_Constraint = tt.evaluate_tt_on_grid(GC_Constraint, H_constraint)
    Y_Obj = tt.evaluate_tt_on_grid(GC_Obj, H_obj)
    S = np.argsort(Y_Obj)

    
    #S = np.argsort(Y_Constraint < constraint_val)
    return Xselected[S[:to_select]], Y_Obj[S[:to_select]], Y_Constraint[S[:to_select]]
