import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from util.loss_torch import bpr_loss, l2_reg_loss
import wandb
from LightGCN import LGCN_Encoder

class ProtoAU(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(ProtoAU, self).__init__(conf, training_set, test_set)
        # init backbone
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

        user_loss = self.proto_loss(user_z, user_prototypes, temperature=temperature)
        item_loss = self.proto_loss(item_z, item_prototypes, temperature=temperature)

        cl_loss = user_loss + item_loss
        return self.config['model_config.proto_reg'] * cl_loss

    def align_loss(x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    def uniform_loss(x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def proto_loss(self, z, prototypes, temperature=0.1):
        # Compute scores between embeddings and prototypes
        # 3862x64 and 2000x64
        scores = torch.mm(z, prototypes.T)

        score_t = scores[: z.size(0) // 2]
        score_s = scores[z.size(0) // 2 :]

        c_t = prototypes(score_t.max(dim=1)[1])
        c_s = prototypes(score_s.max(dim=1)[1])
        loss_au = self.align_loss(c_s, c_t) + self.uniform_loss() # here we choose lambda_3=1

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
        # proto loss is the average of loss_t and loss_s
        proto_loss = (loss_t + loss_s) / 2
        return proto_loss + loss_au

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
