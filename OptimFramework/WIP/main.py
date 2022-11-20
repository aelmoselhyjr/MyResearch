import pandas as pd
import numpy as np
from numpy import ndarray
import surrogate as tt
import optimizers_mask as optim
from pandas import DataFrame
from collections.abc import Callable

#Order of inputs: data, GC specifications, 
#Newton specifications, stochastic specifications, 

def main_loop(constraint_name: str,
              obj_name: str, 
              data_path: str, 
              constraint_val: int, 
              forward_pass: Callable[[ndarray, int, int], ndarray], 
              break_condition: Callable[[ndarray, int, DataFrame], bool],
              
              #GC inputs
              rank: int = 4, 
              max_order_obj: int = 3, 
              max_order_constraint: int = 3, 
              training_loops: int = 50, 
              
              #General optim inputs
              algo_type: str = 'stochastic',
              num_optims: int = 5,
              upper_bound: int = 1,
              lower_bound: int = 0,
              
              #Newton Inputs
              constraint_factor: int = 0.001,
              upper_factor: int = 0.001, 
              lower_factor: int = 0.001) -> None:
    
    """Constrained optimization routine (upper bound)

    Parameters
    ----------
    constraint_name : str,
    obj_name : str, 
    data_path: str, 
    constraint_val: int, 
    forward_pass: Callable[[ndarray, int, int], ndarray], 
    break_condition: Callable[[ndarray, int, DataFrame], bool],
    
    rank: int = 4, 
    max_order_obj: int = 3, 
    max_order_constraint: int = 3, 
    training_loops: int = 50, 
    
    algo_type: str = 'stochastic',
    num_optims: int = 5,
    upper_bound: int = 1,
    lower_bound: int = 0,
    
    constraint_factor: int = 0.001,
    upper_factor: int = 0.001, 
    lower_factor: int = 0.001) -> None:
    
    
    Returns
    -------
    None - appends to data found at data_path

    """
    
    
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
        GC_Constraint, tt_rank, Y, Y_estimated, W = tt.main(constraint_name, total_data, max_order=max_order_constraint, 
                                                      initial_rank=rank, max_rank=rank, 
                                                      numLoops=training_loops)
        num_vars = len(GC_Constraint)
        print("Accuracy of the Constraint fit: Y, Y_Est, Diff")
        print(np.linalg.norm(Y).round(2), np.linalg.norm(Y_estimated).round(2), 
              np.linalg.norm(Y-Y_estimated).round(2))
        print("Y_min, Y_max, Y_est_min, Y_est_max")
        print(Y.min(), Y.max(), Y_estimated.min(), Y_estimated.max())
        
        GC_Obj, tt_rank, Y, Y_estimated, W = tt.main(obj_name, total_data, max_order=max_order_obj, 
                                                      initial_rank=rank, max_rank=rank, 
                                                      numLoops=training_loops)
        print("Accuracy of the Obj fit: Y, Y_Est, Diff")
        print(np.linalg.norm(Y).round(2), np.linalg.norm(Y_estimated).round(2), 
              np.linalg.norm(Y-Y_estimated).round(2))
        print("Y_min, Y_max, Y_est_min, Y_est_max")
        print(Y.min(), Y.max(), Y_estimated.min(), Y_estimated.max())
        if algo_type.lower() == 'newton':
            xopt = optim.gradient_optimize_tt(GC_Constraint, GC_Obj, n = 10, constraint_val = constraint_val, 
                                        max_order_obj = max_order_obj, max_order_constraint = max_order_constraint, 
                                        upper_bound = upper_bound, lower_bound = lower_bound, 
                                        upper_factor = upper_factor, lower_factor = lower_factor, 
                                        constraint_factor = constraint_factor)
            inputs, constraint, value = forward_pass(xopt)
            #blah
        elif algo_type.lower() == 'stochastic':
            print("Optimizing ...")
            xopt, *_ = optim.sampling_optimize_tt(GC_Constraint, GC_Obj, constraint_val, 
                                            max_order_obj = max_order_obj, 
                                            max_order_constraint = max_order_constraint, 
                                            upper_bound = upper_bound, lower_bound = lower_bound,
                                            to_select = num_optims, xopt = xopt)
            # inputs 1x num_vars: optimal
            # constraint: scalar value of constraints at optimal
            # value: scalar value of the objective at optimal
            inputs, constraint, value = forward_pass(xopt)
            
        print("Optimized!")

        
        print("ACTUAL VALUE:", value.round(2))
        # adds new data point to the dataframe
        #new_frame = pd.DataFrame(data = np.array([*inputs, constraint, value]).reshape(1,-1), 
        #                         columns = cols)
        # here is the data updated
        print("Updating data ...")
        #total_data = pd.concat([total_data, new_frame])
        cols=[f'x{i}' for i in range(num_vars)]+['obj', 'cost']
        new_frame = pd.DataFrame(np.concatenate([inputs, value[:,None], constraint[:,None]], axis=1), columns=cols)
        print("Augmenting Frame", new_frame.round(2))
        print("Data Adding!", new_frame.shape)
        total_data = pd.concat([total_data, new_frame], axis=0)
        print("Data Augmenting!", total_data.shape)        
        total_data.to_csv(data_path, index = False)
    
        #print("Paramater Vals:", xopt.round(2))
        print(total_data.sort_values(obj_name)[:5])
        if new_frame[constraint_name].values[0]>constraint_val:
            print("Constraint Violation")
            continue
        elif break_condition(constraint, value, total_data):
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

def toy_func(x):
    print("In Toy function/Showing DIMS", x.shape)
    obj_func = (0.5-x[:,0])**2  + 100*(x[:,1]-x[:,0]**2)**2
    constraint_func = np.nansum(x, axis=-1)
    return x, constraint_func, obj_func

def my_break(constraint, obj, data):
    return False
    if constraint<1e7 and obj < 0.05:
        return True
    else:
        return False
    
    
def init_toy_problem(num_samples = 10000, num_vars = 30):
    #cols = []
    data_path = "./dummy_func_30.csv"
    #data = pd.DataFrame(data = [], columns = cols)
    #for i in range(30):
    #    cols.append("x"+str(i))
    #cols.append("Constraint")
    #cols.append("Obj")
    X = np.random.rand(num_samples, num_vars)
    obj = np.nansum((X - 0.5)**2, axis=-1)
    cost = np.nansum(X, axis=-1)
    cols=[f'x{i}' for i in range(num_vars)]+['obj', 'cost']
    pd.DataFrame(np.concatenate((X, obj[:,None], cost[:,None]), axis=1), columns = cols).to_csv("./dummy_func_30.csv", index = False)
    #for i in range(100000):
    #    inputs, constraint, value = toy_func(np.random.rand(30,))
    #    new_frame = pd.DataFrame(data = np.array([*inputs, constraint, value]).reshape(1,-1), columns = cols)
    #    data = pd.concat([data, new_frame], ignore_index = True)
    
    
def init_toy_problem(num_samples = 10000, num_vars = 2):
    #cols = []
    data_path = "./dummy_func_ros.csv"
    #data = pd.DataFrame(data = [], columns = cols)
    #for i in range(30):
    #    cols.append("x"+str(i))
    #cols.append("Constraint")
    #cols.append("Obj")
    X = np.random.rand(num_samples, num_vars)
    obj = (0.5-X[:,0])**2  + 100*(X[:,1]-X[:,0]**2)**2
    cost = np.nansum(X, axis=-1)
    cols=[f'x{i}' for i in range(num_vars)]+['obj', 'cost']
    pd.DataFrame(np.concatenate((X, obj[:,None], cost[:,None]), axis=1), columns = cols).to_csv("./dummy_func_ros.csv", index = False)
    #for i in range(100000):
    #    inputs, constraint, value = toy_func(np.random.rand(30,))
    #    new_frame = pd.DataFrame(data = np.array([*inputs, constraint, value]).reshape(1,-1), columns = cols)
    #    data = pd.concat([data, new_frame], ignore_index = True)
        
    
if False:
    #init problem
    num_samples = 5000
    num_vars = 30
    init_toy_problem(num_samples, num_vars)
    
    
if False:
    #example run
    num_samples = 5000
    num_vars = 30
    init_toy_problem(num_samples, num_vars)

if False:
    # Rank effectiveness comparison
    total_data = pd.read_csv("./dummy_func_30.csv")
    test_data = pd.read_csv("./test_30.csv")
    
    for rank in range(1, 11):
        print("RANK:", rank)
        GC_Obj, tt_rank, Y, Y_estimated, W = tt.main("obj", total_data, max_order=3, 
                                                              initial_rank=rank, max_rank=rank, 
                                                              numLoops=50)
    
        print("Error on TRAINING:", np.linalg.norm(Y-Y_estimated)/np.linalg.norm(Y)*100)
    
        X = test_data[test_data.columns[:-2].values].values
        Y = test_data["obj"].values
        H = tt.get_orthogonal_polynomial(X, 3)
        Y_estimated = tt.evaluate_tt_on_grid(GC_Obj, H)
        print("Error on Testing:", np.linalg.norm(Y-Y_estimated)/np.linalg.norm(Y)*100)
    
    for rank in range(1, 11):
        print("RANK:", rank)
        GC_Obj, tt_rank, Y, Y_estimated, W = tt.main("cost", total_data, max_order=4, 
                                                              initial_rank=rank, max_rank=rank, 
                                                              numLoops=50)
    
        print("Error on TRAINING:", np.linalg.norm(Y-Y_estimated)/np.linalg.norm(Y)*100)
    
        X = test_data[test_data.columns[:-2].values].values
        Y = test_data["cost"].values
        H = tt.get_orthogonal_polynomial(X, 4)
        Y_estimated = tt.evaluate_tt_on_grid(GC_Obj, H)
        print("Error on Testing:", np.linalg.norm(Y-Y_estimated)/np.linalg.norm(Y)*100)
    print("Done!")







