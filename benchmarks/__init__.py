from typing import Dict, Tuple
from torch.utils.data import DataLoader, random_split
import torch
from tqdm import tqdm
import math
from benchmarks.shape_bias.shape_bias_benchmark import ShapeBiasBenchmark
from datasets import cueconflict_dataloader

class Benchmark:
    """A benchmark is ran for each model and returns a tuple with two
    dictionaries. The first returns the benchmark data, the second returns meta-data.
    """
    def __init__(self,name):
        self.name = name
        
    def __call__(self,model) -> Tuple(Dict,Dict):
        raise NotImplementedError
    
class LayeredBenchmark(Benchmark):
    """A Layered benchmark is a benchmark but runs for some layers of a model
    """
    def __init__(self,name):
        super().__init__(name)
        
    def __call__(self,model,layers:Dict=None) -> Tuple(Dict,Dict):
        raise NotImplementedError
        
class Accuracy(Benchmark):
    def __init__(self,name,dataset,batch_size,k=(1,5)):
        super().__init__(name)
        self.k = k
        self.dataset = dataset
        self.batch_size = batch_size
        
    def __call__(self,model) -> Tuple(Dict,Dict):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataloader = DataLoader(self.dataset,batch_size=self.batch_size,shuffle=True)
        results = [0] * len(self.k)
        total = 0
        model = model.eval().to(device)
        num_of_batches = math.ceil(len(self.dataset)/self.batch_size) 
        bar = tqdm(total=num_of_batches, unit="batches")
        for x,y in dataloader:
            logits = model(x.to(device))
            probs = torch.softmax(logits,dim=1)
            for i in range(len(self.k)):
                results[i] += self._top(probs,y,k=self.k[i])
            total += x.shape[0]
            bar.update()
        assert total == len(dataloader.dataset)
        return (dict((f"top{self.k[i]}_accuracy",results[i]/len(dataloader.dataset)) for i in range(len(self.k))),{})

    def _top(self,probs,labels,k=1):
        sorted_indices = probs.argsort()[:,-k:]
        correct_in_batch = (sorted_indices == labels.view(-1,1)).any(dim=1)
        num_correct = correct_in_batch[correct_in_batch == True].numel()
        return num_correct
    

from benchmarks.shape_bias import ShapeBiasBenchmark