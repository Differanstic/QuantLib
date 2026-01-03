import os
import pandas as pd
from functools import wraps

def log(file='result.csv',input=None):
    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):

            result = func(*args, **kwargs)
            if input:
                data = kwargs[input] | result
            if isinstance(result, tuple):
                data = args[0] | result[0]
            else:
                data = args[0] | result
            if os.path.exists(file):
                df = pd.read_csv(file)
                df= pd.concat([df, pd.DataFrame([data])], ignore_index=True)
                df.to_csv(file,index=False,mode="w",header=True)
            else:
                pd.DataFrame([data]).to_csv(file,index=False,mode="a",header= not os.path.exists(file))

            return result
        return inner
    return decorator