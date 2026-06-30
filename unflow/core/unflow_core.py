from functools import wraps
from unflow.core.engine import ExecutionEngine
from multiprocessing import Pool
from unflow.core.unflow_types import RState, Transformation


REGISTRY : dict = {}
engine = ExecutionEngine()

def _worker(task):
    func_name, args = task
    func = REGISTRY[func_name]
    return engine.run(func, args)


class unflowdecorator:
    
    def __call__(self, func):
        REGISTRY[func.__name__] = func

        @wraps(func)
        def wrapper(*args, **kwargs):
            return engine.run(func, *args, **kwargs)

        wrapper.run_in_parallel = lambda combos: self.parallel(func, combos)
        return wrapper
    
    def parallel(self, func, combinations):
        with Pool() as pool:
            return pool.map(
                _worker,
                [(func.__name__, kwargs) for kwargs in combinations]
            )

   
        
