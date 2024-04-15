import torch
from abc import ABC, abstractmethod
import pdb
import csv
import os

class MetricCalculator(ABC):
    
    @abstractmethod
    def update(self, batch_size, dx, *feats, **aux_params): return NotImplemented
    
    @abstractmethod
    def finalize(self): return NotImplemented

def compute_correlation(covariance, eps=1e-7):
    std = torch.diagonal(covariance).sqrt()
    covariance = covariance / (torch.clamp(torch.outer(std, std), min=eps))
    return covariance

class CovarianceMetric(MetricCalculator):
    name = 'covariance'
    
    def __init__(self):
        self.std = None
        self.mean = None
        self.outer = None
    
    def update(self, batch_size, *feats, **aux_params):
        feats = torch.cat(feats, dim=0)
        feats = torch.nan_to_num(feats, 0, 0, 0)
        
        std = feats.std(dim=1)
        mean = feats.mean(dim=1)
        outer = (feats @ feats.T) / feats.shape[1]
        
        if self.mean  is None: self.mean  = torch.zeros_like( mean)
        if self.outer is None: self.outer = torch.zeros_like(outer)
        if self.std   is None: self.std   = torch.zeros_like(  std)
            
        self.mean  += mean  * batch_size
        self.outer += outer * batch_size
        self.std   += std   * batch_size
    
    def finalize(self, numel, covsave_path, node, eps=1e-4):
        self.outer /= numel
        self.mean  /= numel
        self.std   /= numel
        cov = self.outer - torch.outer(self.mean, self.mean)
        if torch.isnan(cov).any():
            breakpoint()
        if (torch.diagonal(cov) < -0.0001).sum():
            print("Negative COV")
            print(cov)
            
            pdb.set_trace()
        # Check if the output file already exists
        # Check if the specified directory exists, and create it if not
        """output_directory=covsave_path
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Construct the full path for the output file

        output_file = os.path.join(output_directory, f"covariance_{node}.csv")
        
        # Write the covariance matrix to the output file
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in cov.tolist():
                writer.writerow(row)"""
        return cov

class Py_CovarianceMetric(MetricCalculator):
    name = 'py_covariance'
    
    def __init__(self):
        self.feats = None
    
    def update(self, batch_size, *feats, **aux_params):
        feats = torch.cat(feats, dim=0)
        feats = torch.nan_to_num(feats, 0, 0, 0)
        
        if self.feats is None:
            self.feats = feats
        #else:
            #self.feats = torch.cat([self.feats, feats], dim=1)
    
    def finalize(self, numel, covsave_path, node, eps=1e-4):
        # Transpose feats for torch.cov
        feats_transposed = self.feats.T

        # Calculate the covariance matrix using torch.cov
        cov = torch.cov(feats_transposed)

        if torch.isnan(cov).any():
            breakpoint()
        if (torch.diagonal(cov) < 0).sum():
            print("Negative COV")
            print(torch.diagonal(cov))
            output_file = "negative_py_covariance.csv"
            if output_file is not None:
                with open(output_file, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    for row in cov.tolist():
                        writer.writerow(row)
            pdb.set_trace()
        return cov

class CorrelationMetric(MetricCalculator):
    name = 'correlation'
    
    def __init__(self):
        self.std = None
        self.mean = None
        self.outer = None
    
    def update(self, batch_size, dx, *feats, **aux_params):
        feats = torch.cat(feats, dim=0)
        
        std = feats.std(dim=1)
        mean = feats.mean(dim=1)
        outer = (feats @ feats.T) / feats.shape[1]
        
        if self.std   is None: self.std   = torch.zeros_like(  std)
        if self.mean  is None: self.mean  = torch.zeros_like( mean)
        if self.outer is None: self.outer = torch.zeros_like(outer)
            
        self.std   += std   * dx
        self.mean  += mean  * dx
        self.outer += outer * dx
    
    def finalize(self, covsave_path, node, eps=1e-4):
        corr = self.outer - torch.outer(self.mean, self.mean)
        corr /= (torch.outer(self.std, self.std) + eps)
        return corr


class CossimMetric(MetricCalculator):
    name = 'cossim'
    
    def __init__(self):
        self.std = None
        self.mean = None
        self.outer = None
    
    def update(self, batch_size, dx, *feats, **aux_params):
        feats = torch.cat(feats, dim=0)
        
        feats = feats.view(feats.shape[0], -1, batch_size)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.view(feats.shape[0], -1)
        outer = (feats @ feats.T) / (feats.shape[1] // batch_size)
        
        if self.outer is None: self.outer = torch.zeros_like(outer)
            
        self.outer += outer * dx
    
    def finalize(self, covsave_path, node, eps=1e-4):
        return self.outer


class MeanMetric(MetricCalculator):
    name = 'mean'
    
    def __init__(self):
        self.mean = None
    
    def update(self, batch_size, *feats, **aux_params):
        feats = torch.cat(feats, dim=0)
        mean = feats.abs().mean(dim=1)
        if self.mean is None: 
            self.mean = torch.zeros_like(mean)
        self.mean  += mean  * batch_size
    
    def finalize(self, numel, covsave_path, node, eps=1e-4):
        return self.mean / numel
        

def get_metric_fns(names):
    metrics = {}
    for name in names:
        if name == 'mean':
            metrics[name] = MeanMetric
        elif name == 'covariance':
            metrics[name] = CovarianceMetric
        elif name == 'correlation':
            metrics[name] = CorrelationMetric
        elif name == 'cossim':
            metrics[name] = CossimMetric
        else:
            raise NotImplementedError(name)
    return metrics