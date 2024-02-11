import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender

from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss
import wandb


class ProtoAU(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(ProtoAU, self).__init__(conf, training_set, test_set)
        self.model = LGCN_Encoder(
            self.data,
            self.config["embedding_size"],
            self.config["model_config.eps"],
            self.config["model_config.num_layers"],
            self.config["model_config.num_clusters"],
            self.config['model_config.layer_cl']
        )

    def cal_cl_loss(self, idx, user_emb, item_emb, prototypes, temperature):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()

        rec_user_emb, cl_user_emb = user_emb[0], user_emb[1]
        rec_item_emb,cl_item_emb = item_emb[0], item_emb[1]
        user_prototypes, item_prototypes = prototypes[0], prototypes[1]

        user_z = torch.cat([rec_user_emb[u_idx], cl_user_emb[u_idx]], dim=0)
        item_z = torch.cat([rec_item_emb[i_idx], cl_item_emb[i_idx]], dim=0)

        user_loss = self.swav_loss(user_z, user_prototypes, temperature=temperature)
        item_loss = self.swav_loss(item_z, item_prototypes, temperature=temperature)
        cl_loss = user_loss + item_loss
        return self.config['model_config.proto_reg'] * cl_loss

    def swav_loss(self, z, prototypes, temperature=0.1):
        # Compute scores between embeddings and prototypes
        # 3862x64 and 2000x64
        scores = torch.mm(z, prototypes.T)

        score_t = scores[: z.size(0) // 2]
        score_s = scores[z.size(0) // 2 :]

        # Apply the Sinkhorn-Knopp algorithm to get soft cluster assignments
        q_t = self.sinkhorn_knopp(score_t)
        q_s = self.sinkhorn_knopp(score_s)

        log_p_t = torch.log_softmax(score_t / temperature + 1e-7, dim=1)
        log_p_s = torch.log_softmax(score_s / temperature + 1e-7, dim=1)

        # Calculate cross-entropy loss
        loss_t = torch.mean(
            -torch.sum(
                q_s * log_p_t,
                dim=1,
            )
        )
        loss_s = torch.mean(
            -torch.sum(
                q_t * log_p_s,
                dim=1,
            )
        )
        # SwAV loss is the average of loss_t and loss_s
        swav_loss = (loss_t + loss_s) / 2
        return swav_loss

    def sinkhorn_knopp(self, scores, epsilon=0.05, n_iters=3):
        with torch.no_grad():
            scores_max = torch.max(scores, dim=1, keepdim=True).values
            scores_stable = scores - scores_max
            Q = torch.exp(scores_stable / epsilon).t()
            Q /= (Q.sum(dim=1, keepdim=True) + 1e-8)

            K, B = Q.shape
            u = torch.zeros(K).to(scores.device) 
            r = torch.ones(K).to(scores.device) / K
            c = torch.ones(B).to(scores.device) / B

            for _ in range(n_iters):
                u = Q.sum(dim=1) 
                Q *= (r / (u + 1e-8)).unsqueeze(1) 
                Q *= (c / Q.sum(dim=0)).unsqueeze(0) 
                
            Q = (Q / Q.sum(dim=0, keepdim=True)).t() 
            return Q

    def train(self):
        model = self.model.cuda()
        batch_size = self.config["batch_size"]
        num_epochs = self.config["num_epochs"]
        lr = self.config["learning_rate"]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(num_epochs):
            for index, batch in enumerate(next_batch_pairwise(self.data, batch_size)):
                user_idx, pos_idx, neg_idx = batch

                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb, user_prototypes, item_prototypes = model(True)

                user_emb, pos_item_emb, neg_item_emb = (
                    rec_user_emb[user_idx],
                    rec_item_emb[pos_idx],
                    rec_item_emb[neg_idx],
                )

                l2_loss = l2_reg_loss(self.config["lambda"], user_emb, pos_item_emb)
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_loss

                # Contrastive learning loss
                # Swapping assignments between views loss
                cl_loss = torch.zeros_like(rec_loss)

                cl_loss = self.cal_cl_loss([user_idx, pos_idx],[rec_user_emb, cl_user_emb], [rec_item_emb, cl_item_emb], [user_prototypes, item_prototypes], self.config["model_config.temperature"])

                batch_loss = rec_loss + cl_loss

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    normalized_user_prototypes = F.normalize(
                        self.model.prototypes_dict['user_prototypes'], p=2, dim=1)
                    normalized_item_prototypes = F.normalize(
                        self.model.prototypes_dict['item_prototypes'], p=2, dim=1)

                    self.model.prototypes_dict['user_prototypes'] = nn.Parameter(normalized_user_prototypes)
                    self.model.prototypes_dict['item_prototypes'] = nn.Parameter(normalized_item_prototypes)

                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "batch_loss": batch_loss.item(),
                        "rec_loss": rec_loss.item(),
                        "cl_loss": cl_loss.item(),
                    }
                )

                if index % 100 == 0:
                    print(
                        "training:",
                        epoch + 1,
                        "batch",
                        index,
                        "rec_loss:",
                        rec_loss.item(),
                        "cl_loss",
                        cl_loss.item(),
                    )
            with torch.no_grad():
                self.user_emb, self.item_emb, _, _ = model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb, _, _ = self.model()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


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
