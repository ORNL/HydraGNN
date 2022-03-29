import os
import os.path as osp
from ogb_utils import *
from tqdm import tqdm
import torch

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

class OGB_pcqm4mv2(InMemoryDataset):
    def __init__(self, root='./dataset/', transform=None, pre_transform=None):

        self.root = root
        if not osp.isdir(self.root):
            raise ValueError("Directory not found: ", self.root)
        super(OGB_pcqm4mv2, self).__init__(self.root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["pcqm4m_gap.csv"]

    @property
    def processed_file_names(self):
        return ['pcqm4mv2_all.pt']

    def download(self):
        #using pre-downloaded data
        for file in self.raw_paths:
            if not osp.isfile(file):
                raise ValueError("File not found: ", file)
        return
    

    def process(self):
        smileset, valueset, splitflag = datasets_load_gap(self.raw_paths[0])

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smileset))):
            graphdata=generate_graphdata(smileset[i], valueset[i])
            graphdata.split=splitflag[i]
            data_list.append(graphdata)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    dataset = OGB_pcqm4mv2(root='/home/6pz/codes/surrogatemodeling/DESIGN/HydraGNN/examples/ogb/dataset/')
    print(len(dataset))
    print(dataset[0])