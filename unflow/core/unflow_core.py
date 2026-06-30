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

    return engine.run(func, **params)




class unflowdecorator:
    
    def __call__(self, func):
        REGISTRY[func.__name__] = func

        @wraps(func)
        def wrapper(*args, **kwargs):
            return engine.run(func, *args, **kwargs)

        wrapper.run_in_parallel = lambda combos: self.parallel(func, combos)
        wrapper.clear_graph = lambda: self._clear_graph(func)
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
    def _clear_graph(self, func):
        g_name = func.__name__
        engine.db.clear_graph(g_name)

   
        
