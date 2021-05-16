import torch

from collections import defaultdict
from itertools   import combinations
from functools   import reduce

from torch_sparse import SparseTensor
from torch        import Tensor

import networkx as nx

class CliquePooling(torch.nn.Module):
  __all_ops__ = {
      'avg': lambda x: torch.mean(x, dim=0),
      'max': lambda x: torch.max(x, dim=0)[0]
  }
  
  def __init__(self, agg_type: str):
    super(CliquePooling, self).__init__()
    assert agg_type in CliquePooling.__all_ops__, \
      "CliquePooling: agg_type should be either `max` or `avg`."
    self.agg_type  = agg_type
    self.operation = CliquePooling.__all_ops__[agg_type]

  @staticmethod
  def get_nodes_num(edge_index):
    """ Compute the number of nodes contained in graph.
        Args:
            edge_index (torch_geometric.Data): A graph's edges.
        Returns:
            nodes_num: The number of nodes contained in the 
                graph.
    """
    if isinstance(edge_index, Tensor):
      nodes_num = int(edge_index.max()) + 1
    else:
      nodes_num = max(edge_index.size(0), edge_index.size(1))
    return nodes_num

  @staticmethod
  def get_edges(edge_index):
    """ Extracts the edges of the graph.
        Args:
            edge_index (torch_geometric.Data): A graph's connectivity, 
                usually in COO format, but can also be an instance of 
                Sparse Tensor.
        Returns:
            edges: The edges of the graph returned as a tuple 
                as a list containing column and row index.
    """
    if isinstance(edge_index, SparseTensor):
      col, row, _ = edge_index.coo()
      edges = list(zip(row, col))
    elif isinstance(edge_index, Tensor):
      row, col = edge_index
      edges = list(zip(row.tolist(), col.tolist()))
    else:
      edges = edge_index.tolist()
    return edges

  @staticmethod
  def get_neighbours(edge_index, v):
    """ Extracts the neighbors of a node by considering the source node 
        if is v then take all the target nodes with this as source node.
        Args:
            edge_index (torch_geometric.Data): A graph's connectivity, 
                usually in COO format, but can also be an instance of 
                Sparse Tensor.
            v (int): the node's index for which we want to find neighbors. 
        Returns:
            set: containing the neighboring nodes of v
    """
    edges = CliquePooling.get_edges(edge_index)

    return set(
      map(
        lambda p: p[1],
        filter(lambda p: p[0] == v, edges)
      )
    )

  @staticmethod
  def to_networkx(edge_index, nodes_num):
    """ Convert a homogeneous graph to a NetworkX graph.
        Args:
            edge_index (torch_geometric.Data): A graph's connectivity, 
                usually in COO format, but can also be an instance of 
                Sparse Tensor.
            nodes_num (int): the number of nodes in graph.
        Returns:
            G: a NetworkX graph.
    """
    G = nx.Graph()

    G.add_nodes_from(range(nodes_num))

    edges = CliquePooling.get_edges(edge_index)

    for (u,v) in edges:
      G.add_edge(u, v)

    return G
  
  @staticmethod
  def get_maximal_cliques(edge_index, nodes_num):
    """ Search for all maximal cliques in a NetworkX graph.
        Args:
            edge_index (torch_geometric.Data): A graph's connectivity, 
                usually in COO format, but can also be an instance of 
                Sparse Tensor.
            nodes_num (int): the number of nodes in graph.
        Returns:
            list of maximal cliques.
    """
    G = CliquePooling.to_networkx(edge_index, nodes_num)

    return nx.find_cliques(G)

  @staticmethod
  def get_nodes_clusters(edge_index, nodes_num):
    """ Sorts cliques w.r.t their length and filters out the cliques
        whose nodes have been already processed previously.
        Args:
            edge_index (torch_geometric.Data): A graph's connectivity, 
                usually in COO format, but can also be an instance of 
                Sparse Tensor.
            nodes_num (int): the number of nodes in graph.
        Returns:
            (considered_cliques, clusters): the cliques that "survived" the
            filtering and the "fuzzy" clusters indexed by nodes and the
            value is the cluster that they belong to.
    """
    maximal_cliques = \
      CliquePooling.get_maximal_cliques(edge_index, nodes_num)
    sorted_cliques = \
      sorted(maximal_cliques, key=len, reverse=True)

    clusters = defaultdict(list)
    cluster_sz = {}

    considered_cliques = []
    cluster_idx = 0

    for clique in sorted_cliques:
      if all([len(clusters[v]) != 0 for v in clique]):
        # All nodes already covered
        continue
      considered_cliques.append(clique)
      cluster_sz[cluster_idx] = len(clique)

      for v in clique:
        if len(clusters[v]) == 0 or \
            cluster_sz[clusters[v][-1]] == cluster_sz[cluster_idx]:
          clusters[v].append(cluster_idx)
      cluster_idx+=1
      if cluster_idx == nodes_num:
        break

    return considered_cliques, clusters

  @staticmethod
  def get_dual_edges(edge_index, considered_cliques, clusters, nodes_num):
    """ After getting all neighbors of nodes and store them in neighbourhood
        dictionary, introduce new edges between the clusters that are equally
        sized and share nodes. Additionally add the intracluster edges.
        Args:
            edge_index (torch_geometric.Data): A graph's connectivity, 
                usually in COO format, but can also be an instance of 
                Sparse Tensor.
            considered_cliques (list): cliques that still have nodes not 
                considered previously.
            clusters (dictionary): containing the fuzzy clusters. 
            nodes_num (int): the number of nodes in graph.
        Returns:
            edges: the updated list of edges.
    """
    def map_and_unite(func, elems):
      if len(elems) == 0:
        return set()
      return set.union(
        *list(
          map(func, elems)
        )
      )

    neighbourhood = { 
      n: CliquePooling.get_neighbours(edge_index, n) for n in range(nodes_num)
    }

    edges = set()

    # Add `sibling` edges:
    for _, values in clusters.items():
      if len(values) > 1:
        for pair in set(combinations(values, 2)):
          edges.add(pair)

    # Compute the clusters' `siblings`
    siblings = [
       map_and_unite(
           lambda c: set(clusters[c]),
           cluster
       ) 
       for cluster in considered_cliques
    ]

    # Add `intra cluster` edges:
    for idx, clique in enumerate(considered_cliques):
      # Compute union of neighbors in the cluster:
      clique_neighbors = \
        map_and_unite(lambda v: neighbourhood[v], clique)

      # Map clique_neighbors's nodes to new nodes
      # and substruct siblings
      dual_neighbors = map_and_unite(
        lambda x: set(clusters[x]),
        clique_neighbors
      ).difference(siblings[idx])
      
      # Add `intra cluster` edges:
      for v in dual_neighbors:
        edges.add(tuple(sorted([idx, v])))

    return list(edges)

  def forward(self, x, edge_index, edge_attr=None, batch=None, attn=None):
    """ Compute nodes (node features) and edges in the dual graph after 
        the CliquePooling.
        Args:
            x (torch_geometric.Dataset): A graph dataset.
            edge_index (LongTensor): Edge coordinates (usually in form of 
                sparse COO matrix form).
            edge_attr (torch_geometric.Data): The. attributes of the edges.
                Defaults to `None`.
            batch (LongTensor, optional): Batch vector, assigning every node
                to a specific example in the batch. Defaults to `None`.
            attn (torch_geometric.nn): Graph attention. Defaults to `None`.
        Returns:
            (x, dual_edges_t, batch): the node features and the edge-list.
    """
    nodes_num = CliquePooling.get_nodes_num(edge_index)
    x_size    = x.size(0)
  
    if batch is None:
      batch = edge_index.new_zeros(x_size)

    # Cluster the vertices w.r.t the maximal cliques they belong to:
    considered_cliques, clusters = \
      CliquePooling.get_nodes_clusters(edge_index, nodes_num)

    # Compute the dual node features:
    dual_x = torch.stack(
      [
        self.operation(torch.index_select(
            x,
            0,
            torch.tensor(clique, dtype=torch.long)
        )) for clique in considered_cliques
      ]
    )

    # Compute the node features:
    x = torch.stack(
      [
        self.operation(torch.index_select(
            dual_x,
            0,
            torch.tensor(clusters[idx], dtype=torch.long)
        )) for idx in range(x_size)
      ]
    )

    # Compute the dual edges:
    dual_edges = \
      CliquePooling.get_dual_edges(
          edge_index,
          considered_cliques,
          clusters,
          nodes_num
      )

    dual_edges_t = \
      torch.transpose(torch.tensor(dual_edges, dtype=torch.long), 0, 1)

    if edge_attr is not None:
      # TODO:
      pass

    return x, dual_edges_t, None, batch, None

  def __repr__(self):
    return '{}(agg_type={})'.format(self.__class__.__name__, self.agg_type)