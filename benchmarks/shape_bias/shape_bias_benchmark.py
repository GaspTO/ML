from benchmarks.shape_bias.decision_mappings import Imagenet_probabilities_to_16class_names_mapping
import torchvision.models as models
import torch
from tqdm import tqdm
import pandas as pd
import re
from benchmarks.benchmark import Benchmark
from datasets import cueconflict_dataloader
from typing import Dict, Tuple

def argsort(x:list):
    return sorted(range(len(x)), key=x.__getitem__)



class ShapeBiasBenchmark(Benchmark):
    def __init__(self,batch_size,num_workers,device=None,print_results=False,log_intermediate=False):
        """
        args:
            device: the model will be put in device, but everything else will be ran on cpu
        """
        self.dataloader = cueconflict_dataloader(batch_size=batch_size,num_workers=num_workers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.print_results = print_results
        self._log_intermediate = log_intermediate

    def __call__(self,model) -> Tuple[Dict,Dict]:
        """
        return: dictionary with some statistics
        args:
            model: torch model to be evaluated
            print_results: if you want this function to print the return
        """
        model = model.to(self.device).eval()  
        logits, shapes, textures, paths = [], [], [], []
        for x, shape_batch, texture_batch, path_batch in tqdm(self.dataloader):
            with torch.no_grad():
                output = model(x.to(self.device)).cpu()
            logits.append(output)
            paths += path_batch
            shapes += shape_batch
            textures += texture_batch
        logits = torch.concat(logits)
        probabilities = torch.nn.functional.softmax(logits,dim=1)
        label_choices = Imagenet_probabilities_to_16class_names_mapping(probabilities.cpu().numpy())
        filenames = list(map(lambda x:re.sub(".*/.*/","",x) , paths))
        data = pd.DataFrame(data={"label_choice":label_choices[:,0], "shape":shapes, "texture":textures, "filename":filenames})
        if self._log_intermediate:
            self._log_intermediate_calculations(logits,probabilities,filenames)
        data = data[data["shape"] != data["texture"]] #remove data without conflict
        statistics = self._calculate_statistics(data)
        if self.print_results: self._print_statistics(statistics)
        return statistics, {}
        
    def _log_intermediate_calculations(self,logits,probabilities,filenames):
        filenames = pd.DataFrame(data={"filename":filenames})
        logits = pd.DataFrame(logits.numpy())
        logits.join(filenames).to_csv("logits.csv")
        probabilities = pd.DataFrame(probabilities.numpy())
        probabilities.join(filenames).to_csv("probabilities.csv")
        
    def _calculate_statistics(self,data):
        total_shape = (data["label_choice"] == data["shape"]).sum()
        total_texture = (data["label_choice"] == data["texture"]).sum()
        total_shape_or_texture = (total_shape + total_texture)
        shape_bias = total_shape / total_shape_or_texture
        shape_match = total_shape / data.shape[0]
        texture_match = total_texture / data.shape[0]
        return  {"total":data.shape[0],
                 "shape_bias":shape_bias,
                 "shape_match":shape_match,
                 "texture_match":texture_match,
                 "total_shape": total_shape,
                 "total_texture":total_texture,
                 "total_shape_or_texture":total_shape_or_texture}
                    
    def _print_statistics(self,statistics:dict):
        for key,value in statistics.items():
            print(str(key) + ": " + str(value))
            



