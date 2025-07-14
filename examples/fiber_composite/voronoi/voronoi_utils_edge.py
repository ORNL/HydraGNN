import json
import os
import os.path as osp
from typing import Callable, List, Optional
import torch
from torch_geometric.data import (
    Data,
    InMemoryDataset,
)
from torch_geometric.io import read_txt_array

import torch_geometric.transforms as T

import glob
import re

class VoronoiNet(InMemoryDataset):

    def __init__(
        self,
        root: str,
        split: str = 'trainval',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        dataset = [],
        x = torch.tensor([]),
        y = torch.tensor([]),
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        
        self.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self) -> List[str]:
        return ['processed_data.pt']
    
    def process(self) -> None:

        def extract_numbers(filepath):
            match_sve = re.search(r'SVE(\d+)', filepath)
            match_v = re.search(r'v(\d+)', filepath)
            match_m = re.search(r'm(\d+)', filepath)
            sve_number = int(match_sve.group(1)) if match_sve else 0
            v_number = int(match_v.group(1)) if match_v else 0
            m_number = int(match_m.group(1)) if match_m else 0
            return (v_number, sve_number, m_number)
        
        data_list = []
        all_y_values = []  # Collect all y values
        all_x_values = []  # Collect all x values

        v_paths = glob.glob("../dataset/*")
        sorted_v_paths = sorted(v_paths, key=extract_numbers)

        # First pass: collect all y and x values
        for v in sorted_v_paths:
            m_paths = glob.glob(v + "/*")
            sorted_m_paths = sorted(m_paths, key=extract_numbers)
            for m in sorted_m_paths:
                results_path = m + '/AllFld_SVE.csv'
                tensor_y = read_txt_array(results_path,sep=',')
                SVE_paths = m + '/SVE*.csv'
                file_paths = glob.glob(SVE_paths)
                sorted_file_paths = sorted(file_paths, key=extract_numbers)
                
                for i,name in enumerate(sorted_file_paths):
                    tensor = read_txt_array(name,sep=',')
                    tensor = tensor.view(-1, tensor.shape[-1])
                    x = tensor[:, :]
                    # x = torch.cat((tensor[:, :3], tensor[:, 4:13]), dim=1)
                    all_x_values.append(x)
                    
                    if len(tensor_y.shape) == 2:
                        # y = torch.cat((tensor_y[i, 3:5]*1e10, tensor_y[i, 6:9], tensor_y[i, 11].unsqueeze(-1)))
                        y = torch.cat((tensor_y[i, 3:5]*1e10, tensor_y[i, 5:9], tensor_y[i, 11].unsqueeze(-1)))
                        # y = tensor_y[i, 3:5]*1e10 #torch.cat((tensor_y[i, 3:5]*1e10, tensor_y[i, 6:9], tensor_y[i, 11].unsqueeze(-1)))
                        all_y_values.append(y)
                    else:
                        # y = torch.cat((tensor_y[3:5]*1e10, tensor_y[6:9], tensor_y[11].unsqueeze(-1)))
                        y = torch.cat((tensor_y[3:5]*1e10, tensor_y[5:9], tensor_y[11].unsqueeze(-1)))
                        # y = tensor_y[3:5]*1e10 #torch.cat((tensor_y[3:5]*1e10, tensor_y[6:9], tensor_y[11].unsqueeze(-1)))
                        all_y_values.append(y)

        # Stack all values and compute min/max
        all_y_tensor = torch.stack(all_y_values)
        all_x_tensor = torch.cat(all_x_values, dim=0)  # Concatenate along batch dimension
        
        y_min = torch.min(all_y_tensor, dim=0)[0]
        y_max = torch.max(all_y_tensor, dim=0)[0]
        y_range = y_max - y_min
        
        x_min = torch.min(all_x_tensor, dim=0)[0]
        x_max = torch.max(all_x_tensor, dim=0)[0]
        x_range = x_max - x_min

        # Second pass: create normalized data objects
        for v in sorted_v_paths:
            m_paths = glob.glob(v + "/*")
            sorted_m_paths = sorted(m_paths, key=extract_numbers)
            for m in sorted_m_paths:
                results_path = m + '/AllFld_SVE.csv'
                tensor_y = read_txt_array(results_path,sep=',')
                SVE_paths = m + '/SVE*.csv'
                file_paths = glob.glob(SVE_paths)

                sorted_file_paths = sorted(file_paths, key=extract_numbers)
                for i,name in enumerate(sorted_file_paths):
                    edges_file = name.replace('SVE', 'edges_')
                    edges_tensor = read_txt_array(edges_file,sep=',')
                    tensor = read_txt_array(name,sep=',')
                    tensor = tensor.view(-1, tensor.shape[-1])
                    pos = tensor[:, :2]/100
                    # x_raw = tensor[:, :13]
                    x_raw = tensor[:, :] #torch.cat((tensor[:, :3], tensor[:, 4:13]), dim=1)
                    
                    # Normalize x data
                    x = (x_raw - x_min) / x_range
                    x = x[:, 13:] 
                    #x = torch.cat((x[:, :11], x[:, 13:]), dim=1)
                    #x = x_raw
                    
                    if len(tensor_y.shape) == 2:
                        # y = torch.cat((tensor_y[i, 3:5]*1e10, tensor_y[i, 6:9], tensor_y[i, 11].unsqueeze(-1)))
                        y = torch.cat((tensor_y[i, 3:5]*1e10, tensor_y[i, 5:9], tensor_y[i, 11].unsqueeze(-1)))
                        # y = tensor_y[i, 3:5]*1e10
                    else:
                        # y = torch.cat((tensor_y[3:5]*1e10, tensor_y[6:9], tensor_y[11].unsqueeze(-1)))
                        y = torch.cat((tensor_y[3:5]*1e10, tensor_y[5:9], tensor_y[11].unsqueeze(-1)))
                        # y = tensor_y[3:5]*1e10
                    
                    # Normalize y data
                    y = (y - y_min) / y_range
                    
                    data = Data(pos=pos, x=x, y=y)
                    
                    # Apply transformations
                    pre_transform = T.KNNGraph(k=10, force_undirected=True)
                    data = pre_transform(data)
                    distance = T.Distance(norm=False)
                    local_cartesian = T.LocalCartesian(norm=False)
                    data = distance(data)
                    data = local_cartesian(data)
                    # data.edge_attr[:,0] = 1/data.edge_attr[:,0]
                    # data.edge_index = data.edge_index.to(torch.int64)
                    
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    data_list.append(data)
        
        self.dataset = data_list
        # Save the data, normalization parameters, and raw x values for plotting
        torch.save({
            'dataset': data_list,
            'y_min': y_min,
            'y_range': y_range,
            'x_min': x_min[13:],
            'x_range': x_range[13:],
            'all_x_values': all_x_tensor[:, 13:],
            #'x_min': torch.cat((x_min[:11], x_min[13:])),
            #'x_range': torch.cat((x_range[:11], x_range[13:])),
            #'all_x_values': torch.cat((all_x_tensor[:, :11], all_x_tensor[:, 13:]), dim=1),
        }, self.processed_paths[0])

    def load(self, path):
        saved_data = torch.load(path)
        if isinstance(saved_data, tuple):  # Handle old format
            self.dataset, self.y_min, self.y_range = saved_data
        else:  # New dictionary format
            self.dataset = saved_data['dataset']
            self.y_min = saved_data['y_min']
            self.y_range = saved_data['y_range']
            self.x_min = saved_data.get('x_min', None)
            self.x_range = saved_data.get('x_range', None)
            self.all_x_values = saved_data.get('all_x_values', None)

    def to(self, device):
        """Move the dataset and its attributes to the specified device"""
        self.y_min = self.y_min.to(device)
        self.y_range = self.y_range.to(device)
        if hasattr(self, 'x_min') and self.x_min is not None:
            self.x_min = self.x_min.to(device)
        if hasattr(self, 'x_range') and self.x_range is not None:
            self.x_range = self.x_range.to(device)
        self.dataset = [data.to(device) for data in self.dataset]
        if hasattr(self, 'all_x_values') and self.all_x_values is not None:
            self.all_x_values = self.all_x_values.to(device)
        return self

    def len(self):
        return len(self.dataset)
#
    def get(self, idx):
        return self.dataset[idx]
