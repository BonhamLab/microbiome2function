from torch_geometric.nn import MessagePassing


class GraphConv(MessagePassing):

    def __init__(self, aggr = 'sum', *, aggr_kwargs = None, flow = "source_to_target", node_dim = -2, decomposed_layers = 1):
        super().__init__(aggr, aggr_kwargs=aggr_kwargs, flow=flow, node_dim=node_dim, decomposed_layers=decomposed_layers)