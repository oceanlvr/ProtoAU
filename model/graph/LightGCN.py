import torch
import torch.nn as nn
import torch.nn.functional as F
from base.torch_interface import TorchGraphInterface

class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, prototype_num, layer_cl):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.prototype_num = prototype_num
        self.layer_cl = layer_cl
        self.embedding_dict = self._init_model()
        self.prototypes_dict = self._init_prototypes()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def _init_prototypes(self):
        initializer = nn.init.xavier_uniform_
        prototypes_dict = nn.ParameterDict({
            'user_prototypes': nn.Parameter(initializer(torch.empty(self.prototype_num, self.emb_size))),
            'item_prototypes': nn.Parameter(initializer(torch.empty(self.prototype_num, self.emb_size))),
        })
        return prototypes_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        all_embeddings_cl = ego_embeddings
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
            if k==self.layer_cl-1:
                all_embeddings_cl = ego_embeddings
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.data.user_num, self.data.item_num])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embeddings_cl, [self.data.user_num, self.data.item_num])
        user_prototypes = self.prototypes_dict['user_prototypes']
        item_prototypes = self.prototypes_dict['item_prototypes']
        if perturbed:
            return user_all_embeddings, item_all_embeddings,user_all_embeddings_cl, item_all_embeddings_cl,user_prototypes,item_prototypes
        return user_all_embeddings, item_all_embeddings,user_prototypes,item_prototypes
