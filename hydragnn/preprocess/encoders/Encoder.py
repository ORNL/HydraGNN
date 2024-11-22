import pdb
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
from torch.nn import Dropout, Linear, Sequential


class LinearEncoder(torch.nn.Module):
    def __init__(self,emb_dim):
        super().__init__()


















def _learn_embeddings(self):
    if self.embed_node:
        self.node_emb = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
    else:
        self.node_emb = self.register_parameter("node_emb", None)
    if self.use_pos_enc and self.use_struc_enc:
        self.pos_emb = nn.Linear(self.pe_dim, self.hidden_dim, bias=False)
        self.struc_emb = nn.Linear(self.se_dim, self.hidden_dim, bias=False)
        if self.self.embed_node:
            self.node_lin = nn.Linear(3*self.hidden_dim, self.hidden_dim, bias=False)
        else:
            self.node_lin = nn.Linear(2*self.hidden_dim, self.hidden_dim, bias=False)
    elif self.use_pos_enc:
        self.pos_emb = nn.Linear(self.pe_dim, self.hidden_dim, bias=False)
        if self.self.embed_node:
            self.node_lin = nn.Linear(2*self.hidden_dim, self.hidden_dim, bias=False)
        else:
            self.node_lin = self.register_parameter("node_lin", None)
    elif self.use_struc_enc:
        self.struc_emb = nn.Linear(self.se_dim, self.hidden_dim, bias=False)
        if self.self.embed_node:
            self.node_lin = nn.Linear(2*self.hidden_dim, self.hidden_dim, bias=False)
        else:
            self.node_lin = self.register_parameter("node_lin", None)
    else:
        self.pos_emb = self.register_parameter("pos_emb", None)
        self.struc_emb = self.register_parameter("struc_emb", None)
    if self.edge_proc:
        if self.use_edge_attr:
            self.edge_emb = nn.Linear(self.edge_dim, self.hidden_dim, bias=False)
        else:
            self.edge_emb = self.register_parameter("edge_emb", None)
        if self.use_pos_enc and self.use_struc_enc:
            self.edge_pos_emb = nn.Linear(self.pe_dim, self.hidden_dim, bias=False)
            self.edge_struc_emb = nn.Linear(self.se_dim, self.hidden_dim, bias=False)
            if self.use_edge_attr:
                self.edge_lin = nn.Linear(3*self.hidden_dim, self.hidden_dim, bias=False)
            else:
                self.edge_lin = nn.Linear(2*self.hidden_dim, self.hidden_dim, bias=False)
        elif self.use_pos_enc:
            self.edge_pos_emb = nn.Linear(self.pe_dim, self.hidden_dim, bias=False)
            if self.use_edge_attr:
                self.edge_lin = nn.Linear(2*self.hidden_dim, self.hidden_dim, bias=False)
            else:
                self.edge_lin = self.register_parameter("edge_lin", None)
        elif self.use_struc_enc:
            self.edge_struc_emb = nn.Linear(self.se_dim, self.hidden_dim, bias=False)
            if self.use_edge_attr:
                self.edge_lin = nn.Linear(2*self.hidden_dim, self.hidden_dim, bias=False)
            else:
                self.edge_lin = self.register_parameter("edge_lin", None)
        else:
            self.edge_pos_emb = self.register_parameter("edge_pos_emb", None)
            self.edge_struc_emb = self.register_parameter("edge_struc_emb", None)
