#basics
import numpy as np
import torch
from torch_geometric.data import Data
import math

#networkx
from torch_geometric.utils.convert import to_networkx, from_networkx, from_scipy_sparse_matrix
import networkx as nx
import matplotlib.pyplot as plt

#Nets
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F

#scipy
from scipy import sparse


####
class GCNConvM(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConvM, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self.lin(x)

        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
                              norm=norm)

    def message(self, x_i, x_j, norm):
        return x_i + (norm.view(-1, 1)*x_j)

class NetM(torch.nn.Module):
    def __init__(self, dsize):
        super(NetM, self).__init__()
        self.conv1 = GCNConvM(dsize, round(dsize/2))#dataset.num_node_features
        self.conv2 = GCNConvM(round(dsize/2), dsize)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #print("Gconv -> conv1")
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        #print("Gconv -> conv2")
        x = self.conv2(x, edge_index)
        #print("Gconv -> softmax")
        return F.log_softmax(x, dim=1)
        return x


#########=========#####
class LaPool():
    def __init__(self,use_shortestPath = True):
        #Create message passing net
        #self.msgNet = GIN()
        self.msgNet = None
        #self.Gconv = GraphConv(emb_dim,emb_dim)
        #self.Gconv = Net(data)
        self.Gconv = None
        #self.Gconv = GraphConv()
        self.data = None
        self.G = None
        self.is_reducible = True
        self.use_shortest_path = use_shortestPath
        self.reduction_count = 0
        self.edge_batch_count = 0
        self.edge_reduced_batch_count = 0
        
    #Graph
    def getAdjacencyMatrix(self):
        """Generate the adjacency matrix from current graph.

        Args:
            none.
        Returns:
            a numpy array.

        """
        #return nx.adjacency_matrix(self.G)
        return nx.to_numpy_array(self.G)
    
    def __AdjToGraph(self,A):
        return nx.from_numpy_matrix(A)
    
    def __CreateGraph(self,data):
        #return to_networkx(data, edge_attrs=['weight'],remove_self_loops=True)
        #return to_networkx(data, edge_attrs=['edge_weight'],remove_self_loops=True)
        return to_networkx(data, remove_self_loops=True)
    
    def __GraphToTorch(self, G):
        return from_networkx(G)
    
    def __AdjToEdge(self, A):
        return from_scipy_sparse_matrix(A)
    
    def ShowGraph(self, show_labels = True):
        """Show the current graph.

        Args:
            show_labels: if True, shows edge values

        """
        #edge_labels = nx.get_edge_attributes(self.G,'state')
        G = self.__CreateGraph(self.data)
        nx.draw(G, with_labels = show_labels)
        if show_labels:
            nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))
    
    def __Get_shortest_path_length(self,G, src, trgt):
        try:
            shortest_path_nodes = nx.shortest_path(G, source=src, target=trgt)
            return len(shortest_path_nodes)-1
        except:
            return float("inf")
    
    #Special Functions
    def __Sparsemax(self, z):
        '''
        Sparsemax courtesy of:
        Larionov, Michael. “What Is Sparsemax?” 
        Medium, Towards Data Science, 16 Feb. 2020, 
        towardsdatascience.com/what-is-sparsemax-f84c136624e4. 
        '''
        try:
            sum_all_z = sum(z)
            z_sorted = sorted(z, reverse=True)
            k = np.arange(len(z))
            k_array = 1 + k * z_sorted
            z_cumsum = np.cumsum(z_sorted) - z_sorted
            k_selected = k_array > z_cumsum
            k_max = np.where(k_selected)[0].max() + 1
            threshold = (z_cumsum[k_max-1] - 1) / k_max
            return np.maximum(z-threshold, 0)
        except:
            return 0
    
    #Core
    def __Pool(self):
        
        ### Signal intensities computation ###
        edges = self.data.edge_index.tolist() #tensor to list
        edges_temp = edges.copy()
        si_list = [0]*self.data.num_nodes #empty list of signal intensities
        
        x_l0 = np.zeros((self.data.num_nodes, self.data.x.shape[1]))
        for i,e in enumerate(edges[0]):
            next_node = edges[1][i]
            
            #2-norm to get s_i
            x = self.data.x[e] - self.data.x[next_node]
            si_list[e] += math.sqrt(x@x)
            
            
            
        
        ### Dynamic selection of centers ###
        temp_bool_list = [True]*self.data.num_nodes
        
        #compare if current node is greater than neighbors
        #print("Lapool -> _Pool() -> compare if current node is greater than neighbors")
        for i,e in enumerate(edges[0]):
            next_node = edges[1][i]
            
            if (si_list[e] > si_list[next_node]) and temp_bool_list[e] != False :
                temp_bool_list[e] = True 
            else:
                temp_bool_list[e] = False
                continue
            
        
        #output of this section
        V_c = np.where(temp_bool_list)[0] #new centers
        k = len(V_c)  #amount of new centers
        self.is_reducible = True if k>0 else False
        
        if not self.is_reducible:
            #print("NOT FURTHER REDUCIBLE")
            #offset due to batching 
            self.data.edge_index += self.edge_reduced_batch_count
            self.edge_reduced_batch_count += self.data.num_nodes
            
            
            A = self.getAdjacencyMatrix()
            self.data.edge_index, self.data.edge_weight = self.__AdjToEdge(sparse.csr_matrix(A))
        
            #data conversion
            self.data.edge_weight = self.data.edge_weight.type(torch.FloatTensor)
            self.data.edge_index = self.data.edge_index.type(torch.LongTensor)
            return self.data #the graph cannot be further reduced
        
        
        #Update batches
        ttemp = torch.tensor([self.data.batch.detach().numpy()[i] for i in V_c])
        self.data.batch = ttemp
        
        # update embeddings
        #print("Lapool -> _Pool() -> updating embeddings")
        #x_l = self.msgNet.forward(self.data.edge_index,
        #                          self.data.x, 
        #                          self.data.edge_weight)
        
        
        #x_l = self.msgNet(self.data)
        x_l = self.data.x
        x_l = np.array(x_l.detach().numpy())
        
        ### Nodes to cluster mapping ###
        #compute v_f = V\V_c
        nodes_set = set(np.arange(self.data.num_nodes))
        V_f = list(nodes_set - set(V_c))
        
        #compute beta_i
        
        if self.use_shortest_path:
            #print("Lapool -> _Pool() -> shortest paths of ",self.data.num_nodes," nodes with ", len(V_c), " V_c")
            beta_i = np.zeros(self.data.num_nodes)
            for ni, n in enumerate(np.arange(self.data.num_nodes)):
                temp_list = []

                for ci, c in enumerate(V_c):
                    temp_list.append(self.__Get_shortest_path_length(self.G, c, n))
                beta_i[ni] = min(temp_list)
                np.seterr(divide='ignore')
                beta_i = 1/beta_i
                beta_i[beta_i == float("inf")] = 0 #!!!!
        else:
            beta_i = np.ones(self.data.num_nodes)
        
        
        #Affinity matrix (C)
        C = np.zeros((self.data.num_nodes,k))
        for ni in np.arange(C.shape[0]):
            #for vci, vc in enumerate(V_c):
            c_i = [0]*k
            if ni in V_c:
                #Kronecker function
                indx = (V_c.tolist()).index(ni)
                c_i[indx] = 1
            else:
                #sparsemax
                a = np.array(x_l[ni])@(np.transpose(np.array(x_l[V_c])))
                a *= beta_i[ni]
                b = sum(abs(x_l[ni]))*sum(sum(abs(x_l[V_c])))
                c_i = a/b
                c_i = self.__Sparsemax(c_i)
                #print("ni:",ni,"c",c_i)

            C[ni] = c_i
        
        #New Adjacency matrix
        A = self.getAdjacencyMatrix()
        An = np.transpose(C)@A@C
        
        #print("Lapool -> _Pool() -> scipy")
        self.data.edge_index, self.data.edge_weight = self.__AdjToEdge(sparse.csr_matrix(An))
        #self.data.edge_index, _ = self.__AdjToEdge(sparse.csr_matrix(An))
        
        
        #extract weights
        #ttemp = torch.tensor([self.data.edge_weight.detach().numpy()[i] for i in V_c])
        #self.data.edge_weight = ttemp
        
        #print("Lapool -> _Pool() -> new embedding")
        self.data.x = torch.FloatTensor(np.transpose(C))@torch.FloatTensor(x_l)
        
        #print("Lapool -> _Pool() -> new embedding -> Conv")
        #self.data.x = self.Gconv(self.data)
        self.data.x = self.msgNet(self.data)
        
        
        #offset due to batching 
        self.data.edge_index += self.edge_reduced_batch_count
        self.edge_reduced_batch_count += k #offset of number of centroids
        
        #data conversion
        self.data.edge_weight = self.data.edge_weight.type(torch.FloatTensor)
        self.data.edge_index = self.data.edge_index.type(torch.LongTensor)
        #self.data.x = self.data.x.type(torch.DoubleTensor)
        
        
            
    def apply(self, x, edge, weight, batch=None):
        """Generate the adjacency matrix from current graph.

        Args:
            pytorch geometric data type objects.
            x: data.x Node features
            edge: data.edge_index
            weight: data.edge_weight (if any)
            batch: data.batch (if any)
        Returns:
            the coarsened version of the inputs.

        """
        if batch is None:
            batch = torch.zeros(x.size(0)).type(torch.LongTensor)
        
        
        if (weight is None):
            wsize = edge.shape[1]
            weight = torch.tensor([1]*wsize, dtype=torch.float)
            #data.edge_weight = weight
        
        data = Data(x=x, edge_index=edge, edge_weight=weight, batch=batch)
        
        self.msgNet = NetM(data.num_node_features) 
        #self.Gconv = Net(data.num_node_features)
        
        #Prepare for batch processing
        btemp = list(data.batch.detach().numpy())

        sets = set(btemp)
        batches = list(sets)
        row, col = data.edge_index
        #edge_batch = list(data.batch[row].detach().numpy())
        edge_batch_r = list(data.batch[row].detach().numpy())
        edge_batch_c = list(data.batch[col].detach().numpy())

        xxcat = torch.tensor([])
        eecat = torch.tensor([])
        wwcat = torch.tensor([])
        bbcat = torch.tensor([])

        dstop = 0
        dstop_edges = 0
        self.edge_batch_count = 0
        self.edge_reduced_batch_count = 0
        
        
        for i in batches:
            
            xin = np.where(np.array(btemp) == i)
            #ein = np.where(np.array(edge_batch) == i)
            xein_r = np.where(np.array(edge_batch_r) == i)
            xein_c = np.where(np.array(edge_batch_c) == i)
            ein = np.array(np.intersect1d(xein_r, xein_c))
            #print("batch division",ein)
            
            xx = data.x[xin]
            #ee = data.edge_index[:,ein[0]] - self.edge_batch_count
            ee = data.edge_index[:,ein] - self.edge_batch_count
            ww = data.edge_weight[ein]
            bb = torch.tensor([i]*len(xin[0]))
            
            #Pooling
            self.data = Data(x=xx, edge_index=ee, edge_weight=ww, batch=bb)
            self.G = self.__CreateGraph(self.data) 
            self.is_reducible = True
            self.__Pool()
            
            
            #self.ShowGraph(False)
            #Concat
            xxcat = torch.cat((xxcat,self.data.x),0)
            eecat = torch.cat((eecat,self.data.edge_index),1).type(torch.LongTensor)
            wwcat = torch.cat((wwcat,self.data.edge_weight),0).type(torch.FloatTensor)
            bbcat = torch.cat((bbcat,self.data.batch),0).type(torch.LongTensor)
            
            self.edge_batch_count += len(xx[:,0])
        #xxcat = xxcat.type(torch.LongTensor)
        wwcat = wwcat.type(torch.FloatTensor)
        self.data = Data(x=xxcat, edge_index=eecat, edge_weight=wwcat, batch=bbcat)
        return xxcat, eecat, wwcat, bbcat
        ###
        
        
    ######## Bellow here are just debugging functions ####
    def demo_init(self,d, show_labels = True):
        '''
        Just a function to show initial graph
        '''
        #d = self.data
        data = d#Data(x=x, edge_index=edge, edge_weight=weight, batch=batch)
        
        if not('edge_weight' in data):
            wsize = data.edge_index.shape[1]
            weight = torch.tensor([1]*wsize, dtype=torch.float)
            data.edge_weight = weight
            
        #data = d
        g = self.__CreateGraph(data)
        nx.draw(g, with_labels = show_labels)
        if show_labels:
            nx.draw_networkx_edge_labels(g, pos=nx.spring_layout(g))
        
