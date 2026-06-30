from functools import wraps
from unflow.core.engine import ExecutionEngine
from multiprocessing import Pool
from unflow.core.unflow_types import RState, Transformation
import inspect

REGISTRY : dict = {}
engine = ExecutionEngine()

def _worker(task):
    func_name, params = task

    func = REGISTRY[func_name]
    engine = ExecutionEngine()

    return engine.run(func, **params)




class unflowdecorator:
    
    def __call__(self, func):
        REGISTRY[func.__name__] = func

        @wraps(func)
        def wrapper(*args, **kwargs):
            return engine.run(func, *args, **kwargs)

        wrapper.run_in_parallel = lambda combos: self.parallel(func, combos)
        return wrapper
    
    def parallel(self, func, combinations):
        sig = inspect.signature(func)

        normalized = [
            dict(sig.bind(**c).arguments)
            for c in combinations
        ]

        tasks = [
            (func.__name__, params)
            for params in normalized
        ]

        with Pool() as pool:
            return pool.map(_worker, tasks)

   
        
