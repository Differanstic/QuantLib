import sys

def check_var_size(var):
    '''
    Returns:
    MB of Variable
    '''
    size = sys.getsizeof(var)
    return size / 1024 

def free_mem(vars_name:list):
    '''
    Dump Variable
    '''
    import gc
    for var in vars_name:
        if var in globals():
            del globals()[var]
    gc.collect()