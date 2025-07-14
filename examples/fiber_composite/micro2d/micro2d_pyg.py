import json
import os
import os.path as osp
import pickle
from typing import Callable, List, Optional

from tqdm import tqdm

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
)
from torch_geometric.io import read_txt_array

import torch_geometric.transforms as T
from hydragnn.preprocess.utils import RadiusGraphPBC # RadiusGraph
import glob
import re

class FiberNet(InMemoryDataset):

    def __init__(
        self,
        root: str,
        split: str = 'trainval',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        
        self.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self) -> List[str]:
        return ['processed_data.pt']
    
    def process(self) -> None:

        with open('VoidSmall_data.pkl', 'rb') as f:
            VoidSmall_data = pickle.load(f)

        data_list = []
        log10_contrast_ratios = [ -1., -2., -3., 1., 2., 3.]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Define the device

        for VE in tqdm(VoidSmall_data):
            centroids = VE['centroids']
            # Vf = VE['Vf']
            C_coefs = VE['C_coefs_diff']
            # x_vec = VE['x']
            # C_coefs = VE['C_coefs']
            # contrast_ratios = VE['contrast_ratios']

            pos = []

            for point in centroids:
                pos.append([point[0]/256, point[1]/256])
                #x.append([point[0]/256, point[1]/256, Vf, r])
            #y.append(C_coefs[i])

            # graph_data = Data(
            #         pos=torch.tensor(pos, dtype=torch.float),
            #         supercell_size=torch.tensor([1,1,0], dtype=torch.float),
            #         edge_attr = None)
                    #x=torch.tensor(x, dtype=torch.float),
                    #y=torch.tensor(y, dtype=torch.float)
            graph_data = Data(
                    pos=torch.tensor(pos, dtype=torch.float))

            #data = data.to(device) 
            #pre_transform = T.RadiusGraph(r=0.8, loop=False, max_num_neighbors=32)
            # pre_transform = RadiusGraphPBC(r=0.5, loop=False, max_num_neighbors=8,num_workers=256) # 
            pre_transform=T.KNNGraph(k=5,force_undirected=True,num_workers=56)
            graph_data = pre_transform(graph_data)
            distance = T.Distance(norm=False)
            local_cartesian = T.LocalCartesian(norm=False)
            graph_data = distance(graph_data)
            graph_data = local_cartesian(graph_data)
            #if self.pre_filter is not None and not self.pre_filter(data):
            #    continue
            #if self.pre_transform is not None:
            #    data = self.pre_transform(data)

            for i, r in enumerate(log10_contrast_ratios):
                x = []
                # y = []
                for point in centroids:
                    x.append([point[0]/256, point[1]/256, r]) # , Vf
                # y.append(C_coefs[i])
                y = C_coefs[i]
                data = Data(pos=graph_data.pos,
                            edge_index=graph_data.edge_index,
                            edge_attr=graph_data.edge_attr,
                            # x=torch.tensor(x, dtype=torch.float),
                            x=torch.tensor(x, dtype=torch.float),
                            y=torch.tensor(y, dtype=torch.float))
                data_list.append(data)
        self.save(data_list,self.processed_paths[0])
        
#   def len(self):
#       return len(self.dataset)
#   def get(self, idx):
#       return self.dataset[idx]
