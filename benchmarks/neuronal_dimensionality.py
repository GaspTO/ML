from ast import Str
from typing import Dict, List, Tuple
import torch      
from torchvision.models import resnet50
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import DataLoader
import numpy as np
from benchmarks.benchmark import LayeredBenchmark
from tqdm import tqdm
import pickle

class TwoFactor_NeuronalDimensionality(LayeredBenchmark):
    def __init__(self,paired_dataloader,max_batches=None,factors=["shape","texture"],device=None):
        self.dataloader = paired_dataloader
        self.max_batches = -1 if max_batches is None else max_batches        
        self.factors = factors + ["residual"]
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device
            
                
    def __call__(self,model,layers=None)-> Tuple[Dict,Dict]:
        output_dict, factor_list = self.output_dicts(model,layers,15)
        #dims, dims_percent = self.dim_est(output_dict, factor_list)
        dim4, dim4_perc, score4 = self.dim_est_4(output_dict, factor_list)
        dim3, dim3_perc, score3 = self.dim_est_3(output_dict, factor_list)
        dim2, dim2_perc, score2 = self.dim_est_2(output_dict, factor_list)
        dim1, dim1_perc, score1 = self.dim_est_1(output_dict, factor_list)
        dim0, dim0_perc, score0 = self.dim_est_0(output_dict, factor_list)
        return {"dim4_shape":dim4_perc[0], "dim4_texture":dim4_perc[1], "dim4_residual":dim4_perc[2],
                "dim3_shape":dim3_perc[0], "dim3_texture":dim3_perc[1], "dim3_residual":dim3_perc[2],
                "dim2_shape":dim2_perc[0], "dim2_texture":dim2_perc[1], "dim2_residual":dim2_perc[2],
                "dim1_shape":dim1_perc[0], "dim1_texture":dim1_perc[1], "dim1_residual":dim1_perc[2],
                "dim0_shape":dim0_perc[0], "dim0_texture":dim0_perc[1], "dim0_residual":dim0_perc[2],
                "score4_shape":score4[0], "score4_texture":score4[1], "score4_residual":score4[2],
                "score3_shape":score3[0], "score3_texture":score3[1], "score3_residual":score3[2],
                "score2_shape":score2[0], "score2_texture":score2[1], "score2_residual":score2[2],
                "score1_shape":score1[0], "score1_texture":score1[1], "score1_residual":score1[2],
                "score0_shape":score0[0], "score0_texture":score0[1], "score0_residual":score0[2],
                "N":output_dict['example1'].shape[1]}
           
    def output_dicts(self,model,layers,max=None):
        factor_list, output_dict = [], {'example1': [],'example2': []}
        for i, (factor, example1, example2, _ , _, _ , _) in enumerate(tqdm(self.dataloader)):
            example1, example2 = example1.to(self.device), example2.to(self.device)
            output1 = self._extract_features(model,layers,example1)
            output2 = self._extract_features(model,layers,example2)
            factor_list.append(factor.detach().cpu().numpy())
            output_dict['example1'].append(output1.detach().cpu().numpy())
            output_dict['example2'].append(output2.detach().cpu().numpy())
            if i == self.max_batches:
                break
        
        output_dict = {"example1": np.concatenate(output_dict["example1"]),
                       "example2": np.concatenate(output_dict["example2"])}
            
        return output_dict, np.concatenate(factor_list)
    
    def dim_est_4(self,output_dict, factors):
        ''' correct one '''
        residual_index = len(self.factors) - 1
        za = output_dict['example1']
        zb = output_dict['example2']

        za_by_factor = dict()
        zb_by_factor = dict()
        mean_by_factor = dict()
        cov_by_factor = dict()
        score_by_factor = dict() #! the original code
        raw_score_by_factor = dict()

        zall = np.concatenate([za,zb], 0)
        mean = np.mean(zall, 0, keepdims=True)
        var = np.sum(np.mean((zall-mean)*(zall-mean), 0)) #! np.sum(variance of each logit)
        for f in range(len(self.factors)):
            if f != residual_index:
                indices = np.where(factors==f)[0]
                za_by_factor[f] = za[indices]
                zb_by_factor[f] = zb[indices]
                mean_by_factor[f] = 0.5*(np.mean(za_by_factor[f], 0, keepdims=True)+np.mean(zb_by_factor[f], 0, keepdims=True))
                cov_by_factor[f] = np.mean((za_by_factor[f]-mean_by_factor[f])*(zb_by_factor[f]-mean_by_factor[f]), 0)
                
                raw_score_by_factor[f] = np.sum(cov_by_factor[f])
                score_by_factor[f] = raw_score_by_factor[f]/var # variance ratio
                
            else:
                score_by_factor[f] = 1.0

        scores = np.array([score_by_factor[f] for f in range(len(self.factors))])

        # SOFTMAX
        dims, dims_percent = self._softmax_dim(scores,za.shape[1]) 
        return dims, dims_percent, scores
    
    def dim_est_3(self,output_dict, factors):
        residual_index = len(self.factors) - 1
        za = output_dict['example1']
        zb = output_dict['example2']

        za_by_factor = dict()
        zb_by_factor = dict()
        var_a_by_factor = dict()
        var_b_by_factor = dict()
        mean_by_factor = dict()
        cov_by_factor = dict()
        score_by_factor = dict() 
        raw_score_by_factor = dict() 
        
        for f in range(len(self.factors)):
            if f != residual_index:
                indices = np.where(factors==f)[0]
                za_by_factor[f] = za[indices]
                zb_by_factor[f] = zb[indices]
                mean_by_factor[f] = 0.5*(np.mean(za_by_factor[f], 0, keepdims=True)+np.mean(zb_by_factor[f], 0, keepdims=True))
                cov_by_factor[f] = np.mean((za_by_factor[f]-mean_by_factor[f])*(zb_by_factor[f]-mean_by_factor[f]), 0)
                var_a_by_factor[f] = np.mean((za_by_factor[f]-mean_by_factor[f])**2, 0) #!
                var_b_by_factor[f] = np.mean((zb_by_factor[f]-mean_by_factor[f])**2, 0) #!
                
                raw_score_by_factor[f] = np.nansum(cov_by_factor[f]/np.sqrt(var_a_by_factor[f]*var_b_by_factor[f])) #!
                score_by_factor[f] = raw_score_by_factor[f]  / za.shape[-1]
 
            else:
                score_by_factor[f] = 1.0

        scores = np.array([score_by_factor[f] for f in range(len(self.factors))])

        # SOFTMAX
        dims, dims_percent = self._softmax_dim(scores,za.shape[1])      
        return dims, dims_percent, scores
        
    def dim_est_2(self,output_dict, factors):
        residual_index = len(self.factors) - 1
        za = output_dict['example1']
        zb = output_dict['example2']

        za_by_factor = dict()
        zb_by_factor = dict()
        var_a_by_factor = dict()
        var_b_by_factor = dict()
        mean_a_by_factor = dict()
        mean_b_by_factor = dict()
        cov_by_factor = dict()
        score_by_factor = dict() 
        raw_score_by_factor = dict()
        
        for f in range(len(self.factors)):
            if f != residual_index:
                indices = np.where(factors==f)[0]
                za_by_factor[f] = za[indices]
                zb_by_factor[f] = zb[indices]
                mean_a_by_factor[f] = np.mean(za_by_factor[f], 0, keepdims=True)
                mean_b_by_factor[f] = np.mean(zb_by_factor[f], 0, keepdims=True)
                cov_by_factor[f] = np.mean((za_by_factor[f]-mean_a_by_factor[f])*(zb_by_factor[f]-mean_b_by_factor[f]), 0)
                var_a_by_factor[f] = np.mean((za_by_factor[f]-mean_a_by_factor[f])**2, 0) #!
                var_b_by_factor[f] = np.mean((zb_by_factor[f]-mean_b_by_factor[f])**2, 0) #!
                
                raw_score_by_factor[f] = np.nansum(cov_by_factor[f]/np.sqrt(var_a_by_factor[f]*var_b_by_factor[f])) #!
                score_by_factor[f] = raw_score_by_factor[f]  / za.shape[-1]
                
            else:
                score_by_factor[f] = 1.0
                
        scores = np.array([score_by_factor[f] for f in range(len(self.factors))])
        # SOFTMAX
        dims, dims_percent = self._softmax_dim(scores,za.shape[1])    
        return dims, dims_percent, scores
 
                
    def dim_est_1(self,output_dict, factors):
        scores = self.dim_est_0_1_abstract(output_dict,factors)
        dims, dims_percent = self._softmax_dim(scores,output_dict['example1'].shape[1])
        return dims, dims_percent, scores    
    
    def dim_est_0(self,output_dict, factors):
        scores = self.dim_est_0_1_abstract(output_dict,factors)
        dims, dims_percent = self._normalize_dim(scores,output_dict['example1'].shape[1])
        return dims, dims_percent, scores
    
    def dim_est_0_1_abstract(self,output_dict, factors):
        residual_index = len(self.factors) - 1
        za = output_dict['example1']
        zb = output_dict['example2']

        za_by_factor = dict()
        zb_by_factor = dict()
        var_a_by_factor = dict()
        var_b_by_factor = dict()
        mean_a_by_factor = dict()
        mean_b_by_factor = dict()
        cov_by_factor = dict()
        score_by_factor = dict() 
        raw_score_by_factor = dict()
        
        for f in range(len(self.factors)):
            if f != residual_index:
                indices = np.where(factors==f)[0]
                za_by_factor[f] = za[indices]
                zb_by_factor[f] = zb[indices]
                mean_a_by_factor[f] = np.mean(za_by_factor[f], 0, keepdims=True)
                mean_b_by_factor[f] = np.mean(zb_by_factor[f], 0, keepdims=True)
                cov_by_factor[f] = np.mean((za_by_factor[f]-mean_a_by_factor[f])*(zb_by_factor[f]-mean_b_by_factor[f]), 0)
                var_a_by_factor[f] = np.mean((za_by_factor[f]-mean_a_by_factor[f])**2, 0) #!
                var_b_by_factor[f] = np.mean((zb_by_factor[f]-mean_b_by_factor[f])**2, 0) #!
                
                raw_score_by_factor[f] = np.nansum(np.abs(cov_by_factor[f]/np.sqrt(var_a_by_factor[f]*var_b_by_factor[f]))) #!
                score_by_factor[f] = raw_score_by_factor[f]  / za.shape[-1]
                
            else:
                score_by_factor[f] = 1.0
        
        scores = np.array([score_by_factor[f] for f in range(len(self.factors))])
        return scores
        
     
    def _softmax_dim(self,scores,N):
        m = np.max(scores)
        e = np.exp(scores-m)
        softmaxed = e / np.sum(e)
        dim = N
        dims = [int(s*dim) for s in softmaxed]
        dims[-1] = dim - sum(dims[:-1])
        dims_percent = dims.copy()
        for i in range(len(dims)):
            dims_percent[i] = round(100*(dims[i] / sum(dims)),1)
        
        return dims, dims_percent
    
    def _normalize_dim(self,scores,N):
        normalized = scores / scores.sum()
        dim = N
        dims = [int(s*dim) for s in normalized]
        dims[-1] = dim - sum(dims[:-1])
        dims_percent = dims.copy()
        for i in range(len(dims)):
            dims_percent[i] = round(100*(dims[i] / sum(dims)),1)
        
        return dims, dims_percent
        
    def _extract_features(self,model,layers,x):
        feature_extractor = create_feature_extractor(model, return_nodes=layers)
        with torch.no_grad():
            output = feature_extractor(x)      
        return torch.cat(list(output.values())).flatten(1)
    

        
        
        
    