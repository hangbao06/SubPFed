import torch
import random
import numpy as np
import networkx as nx
from collections import defaultdict

import metispy as metis
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_networkx, k_hop_subgraph
from torch_geometric.datasets import Planetoid

import os
from scipy.sparse.csgraph import shortest_path


data_path = '../../datasets'
ratio_train = 0.2
seed = 1234
comms = [1, 2, 3]
n_clien_per_comm = 5

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def generate_data(dataset, n_comms):
    data = split_train(get_data(dataset, data_path), dataset, data_path, ratio_train, 'overlapping', n_comms*n_clien_per_comm)
    split_subgraphs(n_comms, data, dataset)

def split_subgraphs(n_comms, data, dataset):
    G = torch_geometric.utils.to_networkx(data)
    n_cuts, membership = metis.part_graph(G, n_comms)
    assert len(list(set(membership))) == n_comms
    print(f'graph partition done, metis, n_partitions: {len(list(set(membership)))}, n_lost_edges: {n_cuts}')

    adj = to_dense_adj(data.edge_index)[0]
    for comm_id in range(n_comms):
        for client_id in range(n_clien_per_comm):
            client_indices = np.where(np.array(membership) == comm_id)[0]
            client_indices = list(client_indices)
            client_num_nodes = len(client_indices)

            client_indices = random.sample(client_indices, client_num_nodes // 2)
            client_num_nodes = len(client_indices)

            client_edge_index = []
            client_adj = adj[client_indices][:, client_indices]
            client_edge_index, _ = dense_to_sparse(client_adj)
            client_edge_index = client_edge_index.T.tolist()
            client_num_edges = len(client_edge_index)

            client_edge_index = torch.tensor(client_edge_index, dtype=torch.long)
            client_x = data.x[client_indices]
            client_y = data.y[client_indices]
            client_train = data.train[client_indices]
            client_val = data.val_mask[client_indices]
            client_test = data.test[client_indices]

            client_data = Data(
                x = client_x,
                y = client_y,
                edge_index = client_edge_index.t().contiguous(),
                train = client_train,
                val = client_val,
                test = client_test
            )
            assert torch.sum(client_train).item() > 0

            torch_save(data_path, f'{dataset}_overlapping/{n_comms*n_clien_per_comm}/partition_{comm_id*n_clien_per_comm+client_id}.pt', {
                'client_data': client_data,
                'client_id': client_id
            })
            print(f'client_id: {comm_id*n_clien_per_comm+client_id}, iid, n_train_node: {client_num_nodes}, n_train_edge: {client_num_edges}')

for n_comms in comms:
    generate_data(dataset=f'Cora', n_comms=n_comms)
    


def calculate_shortest_paths(adj_matrix):
    """计算所有节点之间的最短路径距离"""
    n_nodes = adj_matrix.shape[0]
    
    # 转换为numpy数组
    if torch.is_tensor(adj_matrix):
        adj_np = adj_matrix.numpy()
    else:
        adj_np = adj_matrix
    
    # 将邻接矩阵转换为距离矩阵（无边则为inf）
    dist_matrix = adj_np.copy().astype(float)
    dist_matrix[dist_matrix == 0] = np.inf
    np.fill_diagonal(dist_matrix, 0)
    
    # 使用Floyd-Warshall算法计算最短路径
    for k in range(n_nodes):
        dist_matrix = np.minimum(dist_matrix, 
                                dist_matrix[:, k:k+1] + dist_matrix[k:k+1, :])
    
    return dist_matrix

def calculate_overlap_distance(subgraph_i, subgraph_j, global_node_mapping_i, global_node_mapping_j):
    """
    计算两个子图之间的基于重叠节点的距离
    
    参数:
    subgraph_i, subgraph_j: 子图的Data对象
    global_node_mapping_i, global_node_mapping_j: 子图节点到原始图节点的映射
    
    返回:
    D_ij: 两个子图之间的距离
    overlap_nodes: 重叠节点集合
    """
    # 获取全局重叠节点
    global_nodes_i = set(global_node_mapping_i.tolist())
    global_nodes_j = set(global_node_mapping_j.tolist())
    overlap_global_nodes = list(global_nodes_i.intersection(global_nodes_j))
    
    if len(overlap_global_nodes) == 0:
        return float('inf'), []
    
    # 创建局部到全局的映射
    local_to_global_i = {local: global_node_mapping_i[local].item() 
                        for local in range(len(global_node_mapping_i))}
    global_to_local_i = {v: k for k, v in local_to_global_i.items()}
    
    local_to_global_j = {local: global_node_mapping_j[local].item() 
                        for local in range(len(global_node_mapping_j))}
    global_to_local_j = {v: k for k, v in local_to_global_j.items()}
    
    # 计算子图的邻接矩阵和距离矩阵
    adj_i = to_dense_adj(subgraph_i.edge_index, max_num_nodes=len(subgraph_i.x))[0]
    adj_j = to_dense_adj(subgraph_j.edge_index, max_num_nodes=len(subgraph_j.x))[0]
    
    dist_i = calculate_shortest_paths(adj_i)
    dist_j = calculate_shortest_paths(adj_j)
    
    # 计算权重（基于重叠节点的度数）
    def calculate_node_weights(subgraph, global_to_local):
        """计算节点权重，基于节点的度数"""
        degrees = torch.sum(adj_i if subgraph == subgraph_i else adj_j, dim=1)
        weights = {}
        for global_node, local_node in global_to_local.items():
            if local_node < len(degrees):
                weights[global_node] = degrees[local_node].item() + 1  # +1避免0权重
        return weights
    
    weights_i = calculate_node_weights(subgraph_i, global_to_local_i)
    weights_j = calculate_node_weights(subgraph_j, global_to_local_j)
    
    # 计算重叠节点的综合权重
    overlap_weights = {}
    for node in overlap_global_nodes:
        if node in weights_i and node in weights_j:
            # 使用两个子图中权重的平均值
            overlap_weights[node] = (weights_i[node] + weights_j[node]) / 2
    
    # 计算距离 D_ij
    numerator = 0.0
    denominator = 0.0
    valid_overlap_nodes = []
    
    for overlap_node in overlap_global_nodes:
        if overlap_node not in overlap_weights:
            continue
            
        weight = overlap_weights[overlap_node]
        
        if overlap_node in global_to_local_i and overlap_node in global_to_local_j:
            local_i = global_to_local_i[overlap_node]
            local_j = global_to_local_j[overlap_node]
            
            # 确保局部节点索引在范围内
            if local_i < dist_i.shape[0] and local_j < dist_j.shape[0]:
                # 计算子图i中所有节点到重叠节点的距离之和
                dist_to_overlap_i = dist_i[:, local_i]
                valid_dist_i = dist_to_overlap_i[np.isfinite(dist_to_overlap_i)]
                sum_dist_i = np.sum(valid_dist_i) if len(valid_dist_i) > 0 else 0
                
                # 计算子图j中所有节点到重叠节点的距离之和
                dist_to_overlap_j = dist_j[:, local_j]
                valid_dist_j = dist_to_overlap_j[np.isfinite(dist_to_overlap_j)]
                sum_dist_j = np.sum(valid_dist_j) if len(valid_dist_j) > 0 else 0
                
                # 累加到分子和分母
                numerator += weight * (sum_dist_i + sum_dist_j)
                denominator += weight * (len(valid_dist_i) + len(valid_dist_j))
                valid_overlap_nodes.append(overlap_node)
    
    # 计算最终距离
    if denominator > 0 and len(valid_overlap_nodes) > 0:
        D_ij = numerator / denominator
    else:
        D_ij = float('inf')
    
    return D_ij, valid_overlap_nodes
