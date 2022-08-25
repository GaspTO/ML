from ast import Str
from typing import Dict, List, Tuple
import torch      
from torchvision.models import resnet50
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import DataLoader
import numpy as np
from benchmarks import LayeredBenchmark

class NeuronalDimensionality(LayeredBenchmark):
    def __init__(self,data:Dict[Tuple[iter,iter]],batch_size,num_workers,device=None):
        self.data = data
        self.batch_size = batch_size
        self.num_workers = num_workers
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device
                
    def __call__(self,model,layers=None)-> Tuple[Dict,Dict]:
        semantic_corrs = {}
        for concept in self.data:
            X,Y = self.data[concept]
            neuronal_corr_sum, _, _ = self._get_correlation_sum(model,X,Y,layers)
            semantic_corrs[concept] = neuronal_corr_sum
        dimensionality = torch.softmax(torch.tensor(list(semantic_corrs.values())),dim=0)
        return {"dimensionality":dimensionality}, {}
           
            
    def _get_correlation_sum(self,model,X,Y,layers=None):
        feature_maps_x, feature_maps_y = self._get_feature_maps(model,X,Y,layers)
        neuronal_corr_map = self._get_neuronal_correlation(feature_maps_x,feature_maps_y,layers)
        total_value, total_neurons, absolute_total_neurons = self._correlation_sum(neuronal_corr_map, layers)
        return total_value, total_neurons, absolute_total_neurons, neuronal_corr_map


    def _get_feature_maps(self,model,X,Y,layers=None):
            if layers is None:
                train_nodes, eval_nodes = get_graph_node_names(model)
                layers = train_nodes
                
            train_feature_extractor = create_feature_extractor(model, return_nodes=layers)
            feature_maps_x = self._extract_features(train_feature_extractor,X)
            feature_maps_y = self._extract_features(train_feature_extractor,Y)
            self._check_feature_maps(X,feature_maps_x)
            self._check_feature_maps(Y,feature_maps_y)
            return feature_maps_x, feature_maps_y    

    def _extract_features(self,feature_extractor,Z):
        feature_map = {}
        for batch in DataLoader(Z,batch_size=1):
            with torch.no_grad():
                feature_map = self._concat_feature_maps(feature_map,feature_extractor(batch))
        return feature_map      
                    
    def _concat_feature_maps(self,feature_map_1, feature_map_2):
        feature_map = {}
        for key in set([*feature_map_1.keys(),*feature_map_2.keys()]):
            if key not in feature_map_1:
                feature_map[key] = feature_map_2[key]
            elif key not in feature_map_2:
                feature_map[key] = feature_map_1[key]
            else:
                feature_map[key] = torch.concat((feature_map_1[key],feature_map_2[key]))
        return feature_map
        
    def _check_feature_maps(self,Z,feature_maps):
        for value in feature_maps.values():
            assert Z.shape[0] == value.shape[0]

    def _get_neuronal_correlation(self,feature_maps_x, feature_maps_y, layers=None):
        if layers is None: layers = feature_maps_x.keys()
        correlation_map = {}
        size = feature_maps_x.values().__iter__().__next__().shape[0]
        for layer in layers:
            neuron_correlations = []
            for neuron in range(size):
                assert feature_maps_x[layer].shape[0] == size and feature_maps_y[layer].shape[0]
                neuronal_values_x = feature_maps_x[layer].view(size,-1)[:,neuron]
                neuronal_values_y = feature_maps_y[layer].view(size,-1)[:,neuron]
                corr = np.corrcoef(torch.stack((neuronal_values_x,neuronal_values_y)))[0][1]
                neuron_correlations.append(corr)
            correlation_map[layer] = neuron_correlations
        return correlation_map
        

    def _correlation_sum(neuronal_corr_map:Dict[Str:List[float]], layers=None) -> Tuple[float,int,int]:
        """Sums all the neuronal correlations of the neurons specified

        Args:
            neuronal_corr_map (Dict[Str:List[float]]): dictionary where layers are the keys
                and the values are lists  where list[n] is the correlation of the nth neuron.
            layers (string): name of layers whose neuronal correlations we are going to sum.
                Defaults to all layers.

        Returns:
            Tuple[float,int,int]: sum of correlations, neurons that were summed, total neurons in the layers considered.
                The two last values are only different if there are some neuronal correlations that are nan.
        """
        if layers is None: layers = neuronal_corr_map.keys()
        neuronal_corr_sum, total_neurons, absolute_total_neurons = 0, 0, 0
        for layer in layers:
            for neuronal_corr in neuronal_corr_map[layer]:
                if not np.isnan(neuronal_corr):
                    neuronal_corr_sum += neuronal_corr
                    total_neurons += 1
                absolute_total_neurons += 1
        return neuronal_corr_sum, total_neurons, absolute_total_neurons
    


