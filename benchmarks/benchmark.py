from typing import Dict, Tuple

class Benchmark:
    """A benchmark is ran for each model and returns a tuple with two
    dictionaries. The first returns the benchmark data, the second returns meta-data.
    """
    def __init__(self,name):
        self.name = name
        
    def __call__(self,model) -> Tuple[Dict,Dict]:
        raise NotImplementedError
    
class LayeredBenchmark(Benchmark):
    """A Layered benchmark is a benchmark but runs for some layers of a model
    """
    def __init__(self,name):
        super().__init__(name)
        
    def __call__(self,model,layers:Dict=None) -> Tuple[Dict,Dict]:
        raise NotImplementedError
        