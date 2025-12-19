import time
import numpy as np
import torch
from scipy.spatial.distance import cosine

from misc.utils import *
from models.nets import *
from modules.federated import ServerModule

class Server(ServerModule):
    def __init__(self, args, sd, gpu_server):
        super(Server, self).__init__(args, sd, gpu_server)
        self.model = GCN(self.args.n_feat, self.args.n_dims, self.args.n_clss, self.args.l1, self.args).cuda(self.gpu_id)
        self.sd['proxy'] = self.get_proxy_data(args.n_feat)
        self.update_lists = []
        self.sim_matrices = []
        
        self.structural_similarity_matrix = None
        self.structural_weight = 0.3 
        self.load_structural_similarity()

    def load_structural_similarity(self):
        try:
            result_path = os.path.join(self.args.data_path, 'Cora_overlapping', f'{self.args.n_clients}', 'result.pt')
            if os.path.exists(result_path):
                result = torch.load(result_path)
                distance_matrix = result['distance_matrix']
                
                max_distance = np.max(distance_matrix[distance_matrix < float('inf')])
                if max_distance > 0:
                    norm_distance = distance_matrix / max_distance
                    self.structural_similarity_matrix = np.exp(-norm_distance * 5)  
                    
                    np.fill_diagonal(self.structural_similarity_matrix, 1.0)
                else:
                    self.structural_similarity_matrix = None
            else:
                self.structural_similarity_matrix = None
                
        except Exception as e:
            self.structural_similarity_matrix = None

    def get_proxy_data(self, n_feat):
        import networkx as nx
        from torch_geometric.utils import from_networkx

        num_graphs, num_nodes = self.args.n_proxy, 100
        data = from_networkx(nx.random_partition_graph([num_nodes] * num_graphs, p_in=0.1, p_out=0, seed=self.args.seed))
        data.x = torch.normal(mean=0, std=1, size=(num_nodes * num_graphs, n_feat))
        return data

    def on_round_begin(self, curr_rnd):
        self.round_begin = time.time()
        self.curr_rnd = curr_rnd
        self.sd['global'] = self.get_weights()

    def on_round_complete(self, updated):
        self.update(updated)
        self.save_state()

    def calculate_comprehensive_similarity(self, functional_sim_matrix, client_indices):
        n_connected = len(client_indices)
        comprehensive_sim_matrix = np.zeros((n_connected, n_connected))
        
        if self.structural_similarity_matrix is None:
            return functional_sim_matrix
        
        structural_sim_submatrix = np.zeros((n_connected, n_connected))
        for i, idx_i in enumerate(client_indices):
            for j, idx_j in enumerate(client_indices):
                if idx_i < self.structural_similarity_matrix.shape[0] and idx_j < self.structural_similarity_matrix.shape[1]:
                    structural_sim_submatrix[i, j] = self.structural_similarity_matrix[idx_i, idx_j]
                else:
                    structural_sim_submatrix[i, j] = 1.0 if i == j else 0.5 
        

        for i in range(n_connected):
            for j in range(n_connected):
                struct_sim = structural_sim_submatrix[i, j]
                func_sim = functional_sim_matrix[i, j]
                
                comprehensive_sim = (self.structural_weight * struct_sim + 
                                   (1 - self.structural_weight) * func_sim)
                
                comprehensive_sim_matrix[i, j] = comprehensive_sim
        
        row_sums = comprehensive_sim_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1 
        comprehensive_sim_matrix = comprehensive_sim_matrix / row_sums[:, np.newaxis]
        
        return comprehensive_sim_matrix

    def update(self, updated):
        st = time.time()
        local_weights = []
        local_functional_embeddings = []
        local_train_sizes = []
        
        client_ids = list(updated.keys()) if isinstance(updated, dict) else updated
        
        for c_id in client_ids:
            if c_id in self.sd:
                local_weights.append(self.sd[c_id]['model'].copy())
                local_functional_embeddings.append(self.sd[c_id]['functional_embedding'])
                local_train_sizes.append(self.sd[c_id]['train_size'])
                del self.sd[c_id]
        
        n_connected = len(local_functional_embeddings)
        
        st = time.time()
        functional_sim_matrix = np.empty(shape=(n_connected, n_connected))
        for i in range(n_connected):
            for j in range(n_connected):
                functional_sim_matrix[i, j] = 1 - cosine(local_functional_embeddings[i], 
                                                         local_functional_embeddings[j])
        
        if self.args.agg_norm == 'exp':
            functional_sim_matrix = np.exp(self.args.norm_scale * functional_sim_matrix)
        
        
        st = time.time()
        comprehensive_sim_matrix = self.calculate_comprehensive_similarity(
            functional_sim_matrix, client_ids
        )
        
        st = time.time()
        fedavg_weights = (np.array(local_train_sizes) / np.sum(local_train_sizes)).tolist()
        
        if self.args.use_comprehensive_agg:
            agg_weights = np.mean(comprehensive_sim_matrix, axis=1)
            agg_weights = agg_weights / np.sum(agg_weights)
            agg_weights = agg_weights.tolist()
        else:
            agg_weights = fedavg_weights
        
        self.set_weights(self.model, self.aggregate(local_weights, agg_weights))
        self.logger.print(f'全局模型已更新 ({time.time()-st:.2f}s)')
        
        st = time.time()
        for i, c_id in enumerate(client_ids):
            personalized_weights = self.aggregate(local_weights, comprehensive_sim_matrix[i, :])
            
            if f'personalized_{c_id}' in self.sd: 
                del self.sd[f'personalized_{c_id}']
            self.sd[f'personalized_{c_id}'] = {'model': personalized_weights}
        
        self.update_lists.append(client_ids)
        self.sim_matrices.append({
            'functional': functional_sim_matrix,
            'comprehensive': comprehensive_sim_matrix
        })
        
        self.logger.print(f'个性化模型已更新 ({time.time()-st:.2f}s)')
        
        self.print_similarity_statistics(functional_sim_matrix, comprehensive_sim_matrix)

    def set_weights(self, model, state_dict):
        set_state_dict(model, state_dict, self.gpu_id)

    def get_weights(self):
        return {
            'model': get_state_dict(self.model),
        }

    def save_state(self):
        torch_save(self.args.checkpt_path, 'server_state.pt', {
            'model': get_state_dict(self.model),
            'sim_matrices': self.sim_matrices,
            'update_lists': self.update_lists,
            'structural_weight': self.structural_weight,
            'structural_similarity_matrix': self.structural_similarity_matrix
        })
    
    def aggregate(self, local_weights, coefficients):
        aggregated_weights = {}
        
        coefficients = np.array(coefficients)
        coefficients = coefficients / np.sum(coefficients)
        
        for key in local_weights[0].keys():
            aggregated_weights[key] = torch.zeros_like(local_weights[0][key])
            
            for w, c in zip(local_weights, coefficients):
                aggregated_weights[key] += c * w[key]
        
        return aggregated_weights