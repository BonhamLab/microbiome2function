from torch_geometric.nn import MessagePassing
from torch.nn.parameter import Parameter
from torch.nn.init import normal_
from torch.nn import Linear, ReLU, Sigmoid, Module, Dropout
from torch import Tensor, cat


class GraphConv(MessagePassing):

    def __init__(self, in_dim: int, msg_dim: int, state_dim: int, aggr = 'max', *, aggr_kwargs = None, flow = "source_to_target", node_dim = -2, decomposed_layers = 1, device=None):
        super().__init__(aggr, aggr_kwargs=aggr_kwargs, flow=flow, node_dim=node_dim, decomposed_layers=decomposed_layers)

        a_data = Tensor(1, device=device)
        normal_(a_data)
        b_data = Tensor([0.0], device=device)
        
        self.gamma = Parameter(a_data)
        self.beta = Parameter(b_data)
        self.msg_lin_transform = Linear(in_dim, msg_dim, bias=False)
        self.upd_lin_transform = Linear(in_dim + msg_dim, state_dim, bias=True)
        self.relu = ReLU()

    def _edge_transform(self, w: Tensor) -> Tensor:
        return self.gamma * w + self.beta
    
    def message(self, h_j: Tensor, edge_attr: Tensor) -> Tensor:
        e_j = self._edge_transform(edge_attr)
        m_j = self.msg_lin_transform(e_j * h_j)
        return m_j
    
    def update(self, aggr_out: Tensor, h: Tensor) -> Tensor:
        pre_h = cat([h, aggr_out], dim=-1)
        h = self.relu(self.upd_lin_transform(pre_h))
        return h

    def forward(self, h, edge_index, edge_attr):
        return self.propagate(h=h, edge_index=edge_index, edge_attr=edge_attr)


class GraphConvNodeClassifier(Module):
    def __init__(self, in_dim: int, msg_dim: int, state_dim: int, out_dim: int):
        super().__init__()
        self.conv1 = GraphConv(in_dim, msg_dim, state_dim)
        self.conv2 = GraphConv(state_dim, msg_dim, state_dim)
        self.lin = Linear(state_dim, out_dim)
        self.relu = ReLU()
        self.dropout = Dropout(p=0.5)
        self.sigmoid = Sigmoid()

    def forward(self, x, edge_index, edge_attr):
        h = self.conv1(x, edge_index, edge_attr)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_attr)
        out = self.lin(h)
        return out
