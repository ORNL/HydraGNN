import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class MultiTaskModelMP:
    def __init__(
        self,
        model: torch.nn.Module,
        group_color: int,
        head_pg: dist.ProcessGroup,
    ):
        self.model = model
        self.shared_pg = dist.group.WORLD
        self.head_pg = head_pg
        self.shared_pg_size = dist.get_world_size(group=self.shared_pg)
        self.shared_pg_rank = dist.get_rank(group=self.shared_pg)
        self.head_pg_size = dist.get_world_size(group=self.head_pg)
        self.head_pg_rank = dist.get_rank(group=self.head_pg)
        print(
            self.shared_pg_rank,
            "shared, head:",
            (self.shared_pg_size, self.shared_pg_rank),
            (self.head_pg_size, self.head_pg_rank),
            group_color,
        )

        assert self.shared_pg_size % self.head_pg_size == 0
        self.total_num_heads = self.shared_pg_size // self.head_pg_size
        self.branch_id = group_color
        print(self.shared_pg_rank, "branch_id:", self.branch_id)

        self.ddp_shared = list()
        self.ddp_head = list()
        for name, layer in model.named_children():
            num_params = sum(p.numel() for p in layer.parameters())
            if num_params == 0:
                continue
            if "heads_NN" in name:
                delete_list = list()
                for i in range(len(layer)):
                    for k in layer[i]:
                        if k not in f"branch-{self.branch_id}":
                            delete_list.append((i, k))

                for i, k in delete_list:
                    print(self.shared_pg_rank, "delete:", i, k)
                    del layer[i][k]

                ddp_module = DDP(layer, process_group=self.head_pg)
                self.ddp_head.append(ddp_module)
            else:
                ddp_module = DDP(layer, process_group=self.shared_pg)
                self.ddp_shared.append(ddp_module)

        ## For compatibility
        self_group = dist.new_group([self.shared_pg_rank])
        self.model = DDP(self.model, process_group=self_group)
        self.module = self.model.module

        return

    def forward(self, x):
        # Forward through the row-replicated shared backbone
        return self.model(x)

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def state_dict(self):
        return self.model.state_dict()

    def train(self):
        for submodel in self.ddp_shared:
            submodel.train()

        for submodel in self.ddp_head:
            submodel.train()

    def eval(self):
        for submodel in self.ddp_shared:
            submodel.eval()

        for submodel in self.ddp_head:
            submodel.eval()

    def __call__(self, x):
        return self.model(x)
