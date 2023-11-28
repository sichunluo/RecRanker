import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE

# Paper: Are graph augmentations necessary? simple graph contrastive learning for recommendation. SIGIR'22


class SimGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set, valid_set):
        super(SimGCL, self).__init__(conf, training_set, test_set, valid_set)
        args = OptionConf(self.config['SimGCL'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.model = SimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers)
        self.model_name =  conf.__getitem__('model.name')
        self.dataset_name =  conf.__getitem__('training.set')[:-13]
        self.aug_type =  conf.__getitem__('aug_type')

    def train(self):

        model = self.model.cuda()

        if self.aug_type == '0':
            load_model=0
            save_model=0
        elif self.aug_type == "1":
            load_model=0
            save_model=1
        elif self.aug_type == "2":
            load_model=1
            save_model=0
        elif self.aug_type == "3":
            load_model=1
            save_model=2

        import os
        if load_model:
            model.load_state_dict(torch.load(f'./model_cpt/{self.model_name}.pt'))
            self.user_emb, self.item_emb = model._get_embedding()
            model._load_model(self.user_emb,self.item_emb)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                if self.aug_type == "2" and n==0:
                    import pandas as pd
                    dff = pd.read_csv(self.dataset_name+f'data_aug_p{self.model_name}.csv')
                    aug_i = [self.data.user[str(_)]  for _ in dff['u'].to_list()]
                    aug_v = [self.data.item[str(_)]  for _ in dff['vv'].to_list()]
                    aug_nv = [self.data.item[str(_)]  for _ in dff['v'].to_list()]
                    user_idx = user_idx+aug_i
                    pos_idx=pos_idx+aug_v
                    neg_idx=neg_idx+aug_nv
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx])
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

        if save_model:
            model._load_model(self.user_emb,self.item_emb)
            save_model_dir = f'./model_cpt/{self.model_name}.pt'
            torch.save(model.state_dict(), save_model_dir)


    def cal_cl_loss(self, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.model(perturbed=True)
        user_view_2, item_view_2 = self.model(perturbed=True)
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)
        return user_cl_loss + item_cl_loss

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class SimGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers):
        super(SimGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings

    def _get_embedding(self):
        return self.embedding_dict.user_emb, self.embedding_dict.item_emb

    def _load_model(self,user_emb, item_emb):
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(user_emb),
            'item_emb': nn.Parameter(item_emb)
        })
        return 