# scad.py
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ---------- 网络结构 ----------
class encoder(nn.Module):
    def __init__(self, n_input, n_latent):
        super(encoder, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        n_hidden = 512
        self.W_1 = nn.Parameter(torch.Tensor(n_hidden, self.n_input).normal_(mean=0.0, std=0.1))
        self.b_1 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))
        self.W_2 = nn.Parameter(torch.Tensor(self.n_latent, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_2 = nn.Parameter(torch.Tensor(self.n_latent).normal_(mean=0.0, std=0.1))

    def forward(self, x):
        h = F.relu(F.linear(x, self.W_1, self.b_1))
        z = F.linear(h, self.W_2, self.b_2)
        return z

class generator(nn.Module):
    def __init__(self, n_input, n_latent):
        super(generator, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        n_hidden = 512
        self.W_1 = nn.Parameter(torch.Tensor(n_hidden, self.n_latent).normal_(mean=0.0, std=0.1))
        self.b_1 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))
        self.W_2 = nn.Parameter(torch.Tensor(self.n_input, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_2 = nn.Parameter(torch.Tensor(self.n_input).normal_(mean=0.0, std=0.1))

    def forward(self, z):
        h = F.relu(F.linear(z, self.W_1, self.b_1))
        x = F.linear(h, self.W_2, self.b_2)
        return x

class discriminator(nn.Module):
    def __init__(self, n_input):
        super(discriminator, self).__init__()
        self.n_input = n_input
        n_hidden = 512
        self.W_1 = nn.Parameter(torch.Tensor(n_hidden, self.n_input).normal_(mean=0.0, std=0.1))
        self.b_1 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))
        self.W_2 = nn.Parameter(torch.Tensor(n_hidden, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_2 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))
        self.W_3 = nn.Parameter(torch.Tensor(1, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_3 = nn.Parameter(torch.Tensor(1).normal_(mean=0.0, std=0.1))

    def forward(self, x):
        h = F.relu(F.linear(x, self.W_1, self.b_1))
        h = F.relu(F.linear(h, self.W_2, self.b_2))
        score = F.linear(h, self.W_3, self.b_3)
        return torch.clamp(score, min=-50.0, max=50.0)

class encoder_site(nn.Module):
    def __init__(self, n_input, n_latent):
        super(encoder_site, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        n_hidd_1, n_hidd_2, n_hidd_3, n_hidd_4 = 1000, 500, 50, 10
        self.fc1 = nn.Linear(n_input, n_hidd_1)
        self.fc1_bn = nn.BatchNorm1d(n_hidd_1)
        self.fc2 = nn.Linear(n_hidd_1, n_hidd_2)
        self.fc2_bn = nn.BatchNorm1d(n_hidd_2)
        self.fc3 = nn.Linear(n_hidd_2, n_hidd_3)
        self.fc3_bn = nn.BatchNorm1d(n_hidd_3)
        self.fc4 = nn.Linear(n_hidd_3, n_hidd_4)
        self.fc4_bn = nn.BatchNorm1d(n_hidd_4)
        self.fc5 = nn.Linear(n_hidd_4, n_latent)

    def forward(self, x):
        h1 = F.relu(self.fc1_bn(self.fc1(x)))
        h2 = F.relu(self.fc2_bn(self.fc2(h1)))
        h3 = F.relu(self.fc3_bn(self.fc3(h2)))
        h4 = F.relu(self.fc4_bn(self.fc4(h3)))
        return self.fc5(h4)

class decoder_site(nn.Module):
    def __init__(self, n_input, n_latent):
        super(decoder_site, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        n_hidd_6, n_hidd_7, n_hidd_8, n_hidd_9 = 10, 50, 500, 1000
        self.fc6 = nn.Linear(n_latent, n_hidd_6)
        self.fc6_bn = nn.BatchNorm1d(n_hidd_6)
        self.fc7 = nn.Linear(n_hidd_6, n_hidd_7)
        self.fc7_bn = nn.BatchNorm1d(n_hidd_7)
        self.fc8 = nn.Linear(n_hidd_7, n_hidd_8)
        self.fc8_bn = nn.BatchNorm1d(n_hidd_8)
        self.fc9 = nn.Linear(n_hidd_8, n_hidd_9)
        self.fc9_bn = nn.BatchNorm1d(n_hidd_9)
        self.fc10 = nn.Linear(n_hidd_9, n_input)

    def forward(self, z):
        h6 = F.relu(self.fc6_bn(self.fc6(z)))
        h7 = F.relu(self.fc7_bn(self.fc7(h6)))
        h8 = F.relu(self.fc8_bn(self.fc8(h7)))
        h9 = F.relu(self.fc9_bn(self.fc9(h8)))
        return self.fc10(h9)


# ---------- 辅助函数 ----------
def get_max_index(vector):
    return np.where(vector == np.max(vector))[0][0]

def trans_plan_b(latent_A, latent_B, metric='correlation', reg=0.1, numItermax=10, device='cpu'):
    cost = ot.dist(latent_A.detach().cpu().numpy(), latent_B.detach().cpu().numpy(), metric=metric)
    cost = torch.from_numpy(cost).float().to(device)
    length_A, length_B = latent_A.shape[0], latent_B.shape[0]
    P = torch.exp(-cost / reg)
    p_s = torch.ones(length_A, 1) / length_A
    p_t = torch.ones(length_B, 1) / length_B
    p_s, p_t = p_s.to(device), p_t.to(device)
    u_s = torch.ones(length_A, 1) / length_A
    u_t = torch.ones(length_B, 1) / length_B
    u_s, u_t = u_s.to(device), u_t.to(device)
    for i in range(numItermax):
        p_t = u_t / torch.mm(torch.transpose(P, 0, 1), p_s)
        p_s = u_s / torch.mm(P, p_t)
    plan = torch.transpose(p_t, 0, 1) * P * p_s
    return plan

def rand_projections(embedding_dim, num_samples=50, device='cpu'):
    projections = [w / np.sqrt((w**2).sum()) for w in np.random.normal(size=(num_samples, embedding_dim))]
    projections = np.asarray(projections)
    return torch.from_numpy(projections).type(torch.FloatTensor).to(device)

def _sliced_wasserstein_distance(encoded_samples, distribution_samples, num_projections=50, p=2, device='cpu'):
    embedding_dim = distribution_samples.size(1)
    projections = rand_projections(embedding_dim, num_projections, device)
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1).to(device))
    distribution_projections = distribution_samples.matmul(projections.transpose(0, 1))
    wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    return wasserstein_distance.mean()

def sliced_wasserstein_distance(encoded_samples, transformed_samples, num_projections=50, p=2, device='cpu'):
    return _sliced_wasserstein_distance(encoded_samples, transformed_samples, num_projections, p, device)


# ---------- 数据拆分工具 ----------
def split_dataset_cv3(N_st, seed=None):
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(N_st)
    np.random.shuffle(indices)
    fold_size = N_st // 3
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(3)]
    results = []
    for i in range(3):
        fold = folds[i]
        half = len(fold) // 2
        val_idx = np.sort(fold[:half])
        test_idx = np.sort(fold[half:])
        train_idx = np.setdiff1d(indices, fold)
        results.append((train_idx.tolist(), val_idx.tolist(), test_idx.tolist()))
    return results

def split_dataset(N_st, seed=None):
    if seed is not None:
        np.random.seed(seed)
    total_indices = np.arange(N_st)
    np.random.shuffle(total_indices)
    train_size = int(2/3 * N_st)
    valid_size = (N_st - train_size) // 2
    training_idx = sorted(total_indices[:train_size])
    validation_idx = sorted(total_indices[train_size:train_size + valid_size])
    test_idx = sorted(total_indices[train_size + valid_size:])
    return training_idx, validation_idx, test_idx


# ---------- 计算 lambda（非一致性度量）----------
def compute_lambda_test(true_coords, z_B, test_idx, k_neighbors=15):
    true_coords = torch.tensor(true_coords, dtype=torch.float)
    z_B = torch.tensor(z_B, dtype=torch.float)
    coords_sub = true_coords[test_idx, :]
    z_sub = z_B[test_idx, :]
    dist_matrix_spatial = torch.cdist(coords_sub, coords_sub)
    dist_matrix_latent = torch.cdist(z_sub, z_sub)
    N = dist_matrix_spatial.shape[0]
    lambda_test = np.zeros((N, 1))
    for i in range(N):
        dists = dist_matrix_spatial[i].clone()
        dists[i] = float('inf')
        dists_k, indices_k = torch.topk(dists, k_neighbors, largest=False)
        lambda_test[i, 0] = dist_matrix_latent[i, indices_k].mean().item()
    return lambda_test


# ---------- 核心模型类 ----------
class Model3(object):
    def __init__(self,
                 beta=1.0,
                 lambda_Lasso=0.1,
                 resolution="low",
                 batch_size=500,
                 train_epoch=5000,
                 cut_steps=0.5,
                 seed=1234,
                 npcs=30,
                 n_latent=20,
                 n_coord=2,
                 sf_coord=12000,
                 location="spatial",
                 rad_cutoff=None,
                 lambdaGAN=1.0,
                 lambdacos=20.0,
                 lambdaAE=10.0,
                 lambdaLA=10.0,
                 lambdaSWD=5.0,
                 lambdalat=1.0,
                 lambdarec=0.01,
                 model_path="models",
                 data_path="data",
                 result_path="results",
                 ot=True,
                 verbose=True,
                 device="cpu"):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        if resolution not in ['low', 'high']:
            raise ValueError("Resolution must be 'low' or 'high'.")
        if resolution == "high" and not ot:
            raise ValueError("If resolution is 'high', ot must be True.")
        self.resolution = resolution
        self.batch_size = batch_size
        self.train_epoch = train_epoch
        self.cut_steps = cut_steps if resolution == "low" else 1
        self.npcs = npcs
        self.n_latent = n_latent
        self.n_coord = n_coord
        self.sf_coord = sf_coord
        self.location = location
        self.rad_cutoff = rad_cutoff
        self.beta = beta
        self.lambda_Lasso = lambda_Lasso
        self.lambdaGAN = lambdaGAN
        self.lambdacos = lambdacos
        self.lambdaAE = lambdaAE
        self.lambdaLA = lambdaLA
        self.lambdaSWD = lambdaSWD
        self.lambdalat = lambdalat
        self.lambdarec = lambdarec
        self.margin = 5.0
        self.model_path = model_path
        self.data_path = data_path
        self.result_path = result_path
        self.ot = ot
        self.verbose = verbose
        self.device = device

    def preprocess(self, svg_list, adata_A_input, adata_B_input, hvg_num=4000, save_embedding=False, res=1.0):
        adata_A_input.obs["batch"] = "scRNA-seq"
        adata_B_input.obs["batch"] = "ST"
        adata_A = adata_A_input.copy()
        adata_B = adata_B_input.copy()
        if self.verbose:
            print("Finding highly variable genes...")
        sc.pp.highly_variable_genes(adata_A, flavor='seurat_v3', n_top_genes=hvg_num)
        sc.pp.highly_variable_genes(adata_B, flavor='seurat_v3', n_top_genes=hvg_num)
        hvg_A = adata_A.var[adata_A.var.highly_variable == True].sort_values(by="highly_variable_rank").index
        hvg_B = adata_B.var[adata_B.var.highly_variable == True].sort_values(by="highly_variable_rank").index
        hvg_total = hvg_A.intersection(hvg_B)
        print('# overlap highly variable genes is: '+str(len(hvg_total)))
        if len(hvg_total) < 100:
            raise ValueError(f"Total highly variable genes <100 ({len(hvg_total)}). Increase hvg_num.")
        if self.verbose:
            print("Normalizing and scaling...")
        sc.pp.normalize_total(adata_A, target_sum=1e4)
        sc.pp.log1p(adata_A)
        svg_list = list(set(svg_list) & set(adata_A.var.index))
        svg_list = list(set(svg_list) & set(adata_B.var.index))
        adata_A_svg = adata_A[:, svg_list].copy()
        sc.pp.scale(adata_A_svg, max_value=10)
        self.emb_svg_A = adata_A_svg.X.copy()
        adata_A0 = adata_A.copy()
        sc.tl.pca(adata_A0)
        sc.pp.neighbors(adata_A0)
        sc.tl.umap(adata_A0)
        sc.tl.leiden(adata_A0, resolution=res)
        cluster_labels = adata_A0.obs['leiden'].astype(int).values
        K = len(set(cluster_labels))
        adata_A = adata_A[:, hvg_total]
        sc.pp.scale(adata_A, max_value=10)
        sc.pp.normalize_total(adata_B, target_sum=1e4)
        sc.pp.log1p(adata_B)
        print(adata_B)
        print(svg_list)
        adata_B_svg = adata_B[:, svg_list].copy()
        sc.pp.scale(adata_B_svg, max_value=10)
        self.emb_svg_B = adata_B_svg.X.copy()
        self.n_svg = adata_B_svg.X.shape[1]
        adata_B = adata_B[:, hvg_total]
        sc.pp.scale(adata_B, max_value=10)
        adata_total = adata_A.concatenate(adata_B, index_unique=None)
        if self.verbose:
            print("Dimensionality reduction via PCA...")
        pca = PCA(n_components=self.npcs, svd_solver="arpack", random_state=0)
        adata_total.obsm["X_pca"] = pca.fit_transform(adata_total.X)
        self.emb_A = adata_total.obsm["X_pca"][:adata_A.shape[0], :self.npcs].copy()
        self.emb_B = adata_total.obsm["X_pca"][adata_A.shape[0]:, :self.npcs].copy()
        temp = adata_B.obsm[self.location][:, :self.n_coord].copy()
        self.coord_B = (temp - temp.min(axis=0)) / (temp.max(axis=0) - temp.min(axis=0))
        self.adata_total = adata_total
        self.adata_A_input = adata_A_input
        self.adata_B_input = adata_B_input
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        if save_embedding:
            np.save(os.path.join(self.data_path, "lowdim_A.npy"), self.emb_A)
            np.save(os.path.join(self.data_path, "lowdim_B.npy"), self.emb_B)
        self.K = K
        self.cluster_labels = cluster_labels
        return K, cluster_labels

    def train(self, training_idx_rna, training_idx_st, num_projections=500, metric='correlation', reg=0.1, numItermax=10):
        begin_time = time.time()
        if self.verbose:
            print("Begining time: ", time.asctime(time.localtime(begin_time)))
        self.E_A = encoder(self.npcs, self.n_latent).to(self.device)
        self.E_B = encoder(self.npcs, self.n_latent).to(self.device)
        self.G_A = generator(self.npcs, self.n_latent).to(self.device)
        self.G_B = generator(self.npcs, self.n_latent).to(self.device)
        self.D_A = discriminator(self.npcs).to(self.device)
        self.D_B = discriminator(self.npcs).to(self.device)
        self.E_s = encoder_site(self.n_latent + self.n_svg, self.n_coord).to(self.device)
        x_A = torch.from_numpy(self.emb_A).float().to(self.device)
        z_A = self.E_A(x_A)
        df = pd.DataFrame(z_A.cpu().detach().numpy())
        df.index = self.cluster_labels
        mean_values = df.groupby(df.index).mean().values
        self.phi = nn.Parameter(torch.ones(self.K, device=self.device) / self.K)
        self.mu = nn.Parameter(torch.tensor(mean_values, device=self.device))
        self.sigma = nn.Parameter(torch.eye(self.n_latent, device=self.device).unsqueeze(0).repeat(self.K, 1, 1))

        params_G = list(self.E_A.parameters()) + list(self.E_B.parameters()) + list(self.G_A.parameters()) + \
                   list(self.G_B.parameters()) + [self.mu, self.phi]
        params_S = list(self.E_s.parameters())
        optimizer_G = optim.Adam(params_G, lr=0.001, weight_decay=0.)
        optimizer_S = optim.Adam(params_S, lr=0.001, weight_decay=0.)
        params_D = list(self.D_A.parameters()) + list(self.D_B.parameters())
        optimizer_D = optim.Adam(params_D, lr=0.001, weight_decay=0.)
        self.E_A.train(); self.E_B.train(); self.G_A.train(); self.G_B.train()
        self.E_s.train(); self.D_A.train(); self.D_B.train()

        def gmm_log_likelihood(z):
            z = z.unsqueeze(1)
            diff = z - self.mu
            cov_inv = torch.stack([torch.inverse(s) for s in self.sigma])
            exponent = -0.5 * torch.einsum('bki,kij,bkj->bk', diff, cov_inv, diff)
            log_prob = exponent - 0.5 * torch.logdet(self.sigma).unsqueeze(0) + torch.log(self.phi)
            return -torch.logsumexp(log_prob, dim=1).mean()

        N_A, N_B = len(training_idx_rna), len(training_idx_st)
        if self.ot:
            plan = np.ones((N_A, N_B)) / (self.batch_size * self.batch_size)
            plan = torch.from_numpy(plan).float().to(self.device)
        emb_svg_B = torch.from_numpy(self.emb_svg_B).float().to(self.device)

        for j in range(self.train_epoch):
            index_A = np.random.choice(training_idx_rna, size=self.batch_size)
            index_B = np.random.choice(training_idx_st, size=self.batch_size)
            x_A = torch.from_numpy(self.emb_A[index_A, :]).float().to(self.device)
            x_B = torch.from_numpy(self.emb_B[index_B, :]).float().to(self.device)
            c_B = torch.from_numpy(self.coord_B[index_B, :]).float().to(self.device)
            z_A = self.E_A(x_A)
            z_B = self.E_B(x_B)
            z_B1 = torch.cat([emb_svg_B[index_B, :], z_B.detach()], dim=1)
            m_B = self.E_s(z_B1)
            x_AtoB = self.G_B(z_A)
            x_BtoA = self.G_A(z_B)
            x_Arecon = self.G_A(z_A)
            x_Brecon = self.G_B(z_B)
            z_AtoB = self.E_B(x_AtoB)
            z_BtoA = self.E_A(x_BtoA)

            if j < int(self.train_epoch * self.cut_steps):
                optimizer_D.zero_grad()
                if j <= 5:
                    loss_D_A = (torch.log(1 + torch.exp(-self.D_A(x_A))) + torch.log(1 + torch.exp(self.D_A(x_BtoA)))).mean()
                    loss_D_B = (torch.log(1 + torch.exp(-self.D_B(x_B))) + torch.log(1 + torch.exp(self.D_B(x_AtoB)))).mean()
                else:
                    loss_D_A = (torch.log(1 + torch.exp(-torch.clamp(self.D_A(x_A), -self.margin, self.margin))) +
                                torch.log(1 + torch.exp(torch.clamp(self.D_A(x_BtoA), -self.margin, self.margin)))).mean()
                    loss_D_B = (torch.log(1 + torch.exp(-torch.clamp(self.D_B(x_B), -self.margin, self.margin))) +
                                torch.log(1 + torch.exp(torch.clamp(self.D_B(x_AtoB), -self.margin, self.margin)))).mean()
                loss_D = loss_D_A + loss_D_B
                loss_D.backward(retain_graph=True)
                optimizer_D.step()

                loss_AE_A = torch.mean((x_Arecon - x_A) ** 2)
                loss_AE_B = torch.mean((x_Brecon - x_B) ** 2)
                loss_AE = loss_AE_A + loss_AE_B
                loss_cos_A = (1 - torch.sum(F.normalize(x_AtoB, p=2) * F.normalize(x_A, p=2), 1)).mean()
                loss_cos_B = (1 - torch.sum(F.normalize(x_BtoA, p=2) * F.normalize(x_B, p=2), 1)).mean()
                loss_cos = loss_cos_A + loss_cos_B
                loss_LA_AtoB = torch.mean((z_A - z_AtoB) ** 2)
                loss_LA_BtoA = torch.mean((z_B - z_BtoA) ** 2)
                loss_LA = loss_LA_AtoB + loss_LA_BtoA
                loss_SWD = sliced_wasserstein_distance(z_A, z_B, num_projections=num_projections, device=self.device)
                if self.ot:
                    plan_tmp = trans_plan_b(z_A, z_B, metric=metric, reg=reg, numItermax=numItermax, device=self.device)
                    coord_list = np.array([[i, j] for i in index_A for j in index_B])
                    plan[coord_list[:, 0], coord_list[:, 1]] = plan_tmp.reshape([self.batch_size * self.batch_size, ])

                optimizer_G.zero_grad()
                loss_GMM = gmm_log_likelihood(z_A)
                # Lasso deconvolution
                D, N = z_B.T.shape
                K = self.mu.shape[0]
                pi_all = np.zeros((N, K))
                X = self.mu.detach().cpu().numpy().T
                lasso = Lasso(alpha=0.01, fit_intercept=False, max_iter=1000)
                z = z_B.detach().cpu().numpy().T
                for i in range(N):
                    y = z[:, i]
                    lasso.fit(X, y)
                    pi_all[i] = lasso.coef_
                pi = torch.from_numpy(pi_all).float().to(self.device)
                spot_emb_comb = torch.mm(pi, self.mu)
                loss_deconv_lin = torch.mean(torch.norm(z_B - spot_emb_comb, dim=1)**2) / (2 * self.n_latent)
                loss_deconv_l1 = torch.mean(torch.norm(pi, p=1, dim=1))
                loss_deconv = loss_deconv_lin + self.lambda_Lasso * loss_deconv_l1
                loss_G = 0.1 * loss_GMM + self.beta * loss_deconv + self.lambdacos * loss_cos + \
                         self.lambdaAE * loss_AE + self.lambdaLA * loss_LA + self.lambdaSWD * loss_SWD
                loss_G.backward()
                optimizer_G.step()

                if not j % 500 and self.verbose:
                    print(pi.shape)
                    print(f"step {j}, total_loss={loss_G.item():.4f}, loss_D={loss_D.item():.4f}, loss_GAN=0, "
                          f"loss_AE={loss_AE.item():.4f}, loss_cos={loss_cos.item():.4f}, "
                          f"loss_LA={loss_LA.item():.4f}, loss_SWD={loss_SWD.item():.4f}")

            else:
                optimizer_S.zero_grad()
                loss_lat = torch.nn.L1Loss()(m_B, c_B) + 0.1 * sliced_wasserstein_distance(m_B, c_B, num_projections=num_projections, device=self.device)
                loss_S = self.lambdalat * loss_lat
                loss_S.backward()
                optimizer_S.step()
                if not j % 500 and self.verbose:
                    print(f"step {j}, loss_lat={loss_lat.item():.4f}")

        torch.save(self.sigma.detach(), os.path.join(self.result_path, 'sigma.pt'))
        torch.save(self.mu.detach(), os.path.join(self.result_path, 'mu.pt'))
        torch.save(self.phi.detach(), os.path.join(self.result_path, 'phi.pt'))
        end_time = time.time()
        if self.verbose:
            print("Ending time: ", time.asctime(time.localtime(end_time)))
            print(f"Training takes {end_time - begin_time:.2f} seconds")
        if self.ot:
            self.plan = plan.detach().cpu().numpy()
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        state = {'D_A': self.D_A.state_dict(), 'D_B': self.D_B.state_dict(),
                 'E_A': self.E_A.state_dict(), 'E_B': self.E_B.state_dict(),
                 'G_A': self.G_A.state_dict(), 'G_B': self.G_B.state_dict(),
                 'E_s': self.E_s.state_dict()}
        torch.save(state, os.path.join(self.model_path, "ckpt.pth"))
        np.save(os.path.join(self.model_path, "pi.npy"), pi.detach().cpu().numpy())

    def eval2(self, D_score=False, save_embedding=False, hvg_num=4000, retain_prop=1):
        self.E_A = encoder(self.npcs, self.n_latent).to(self.device)
        self.E_B = encoder(self.npcs, self.n_latent).to(self.device)
        self.G_A = generator(self.npcs, self.n_latent).to(self.device)
        self.G_B = generator(self.npcs, self.n_latent).to(self.device)
        self.E_s = encoder_site(self.n_latent + self.n_svg, self.n_coord).to(self.device)
        ckpt = torch.load(os.path.join(self.model_path, "ckpt.pth"))
        self.E_A.load_state_dict(ckpt['E_A'])
        self.E_B.load_state_dict(ckpt['E_B'])
        self.G_A.load_state_dict(ckpt['G_A'])
        self.G_B.load_state_dict(ckpt['G_B'])
        self.E_s.load_state_dict(ckpt['E_s'])
        emb_svg_B = torch.from_numpy(self.emb_svg_B).float().to(self.device)
        emb_svg_A = torch.from_numpy(self.emb_svg_A).float().to(self.device)
        x_A = torch.from_numpy(self.emb_A).float().to(self.device)
        x_B = torch.from_numpy(self.emb_B).float().to(self.device)
        z_A = self.E_A(x_A)
        z_B = self.E_B(x_B)
        z_B1 = torch.cat([emb_svg_B, z_B.detach()], dim=1)
        z_A1 = torch.cat([emb_svg_A, z_A.detach()], dim=1)
        m_A = self.E_s(z_A1)
        m_B = self.E_s(z_B1)
        x_AtoB = self.G_B(z_A)
        x_BtoA = self.G_A(z_B)
        mu = torch.load(os.path.join(self.result_path, 'mu.pt')).cpu().numpy()
        sigma = torch.load(os.path.join(self.result_path, 'sigma.pt')).cpu().numpy()
        phi = torch.load(os.path.join(self.result_path, 'phi.pt')).cpu().numpy()

        if self.resolution == "high":
            retain_cell = retain_prop * self.plan.T.shape[1]
            if retain_cell < 5:
                raise ValueError(f"retain_prop too small, retain_cell={retain_cell}")

        self.latent = np.concatenate((z_A.detach().cpu().numpy(), z_B.detach().cpu().numpy()), axis=0)
        self.adata_total.obsm['latent'] = self.latent
        self.data_Aspace = np.concatenate((self.emb_A, x_BtoA.detach().cpu().numpy()), axis=0)
        self.data_Bspace = np.concatenate((x_AtoB.detach().cpu().numpy(), self.emb_B), axis=0)

        if self.resolution == "low":
            self.map = np.concatenate((m_A.detach().cpu().numpy(), m_B.detach().cpu().numpy()), axis=0) * self.sf_coord
            self.map_A = m_A.detach().cpu().numpy() * self.sf_coord
            self.map_B = m_B.detach().cpu().numpy() * self.sf_coord

        self.adata_total.obs["batch"] = self.adata_total.obs["batch"].replace(["0", "1"], ["scRNA-seq", "ST"])
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        if self.resolution == "low":
            self.adata_total.obsm['loc'] = self.map
            self.adata_B_input.obsm["loc"] = self.map_B
            self.adata_A_input.obsm["loc"] = self.map_A
        self.adata_total.write(os.path.join(self.result_path, "adata_total.h5ad"))

        sc.pp.highly_variable_genes(self.adata_A_input, flavor="seurat_v3", n_top_genes=hvg_num)
        sc.pp.normalize_total(self.adata_A_input, target_sum=1e4)
        sc.pp.log1p(self.adata_A_input)
        self.adata_A_input.write(os.path.join(self.result_path, "adata_sc.h5ad"))

        sc.pp.highly_variable_genes(self.adata_B_input, flavor="seurat_v3", n_top_genes=hvg_num)
        sc.pp.normalize_total(self.adata_B_input, target_sum=1e4)
        sc.pp.log1p(self.adata_B_input)
        self.adata_B_input.write(os.path.join(self.result_path, "adata_ST.h5ad"))

        if self.resolution == "low":
            dist_with_spot = cdist(self.adata_A_input.obsm["loc"], self.adata_B_input.obsm["spatial"])
            min_dist = np.min(dist_with_spot, axis=1)
            self.adata_A_keep = self.adata_A_input[min_dist <= self.rad_cutoff]
            if 'spatial' in self.adata_B_input.uns:
                self.adata_A_keep.uns["spatial"] = self.adata_B_input.uns["spatial"]
                self.adata_A_keep.obsm["spatial"] = self.adata_A_keep.obsm["loc"]
            self.adata_A_keep.write(os.path.join(self.result_path, "adata_sc_keep.h5ad"))
            if self.verbose:
                print("Localized scRNA-seq dataset has been saved!")

        if self.ot:
            self.plan_df = pd.DataFrame(self.plan, index=self.adata_A_input.obs.index,
                                        columns=self.adata_B_input.obs.index)
            self.plan_df.index = self.plan_df.index.values
            self.plan_df.to_csv(os.path.join(self.result_path, "trans_plan.csv"))

        self.latent_df = pd.DataFrame(self.latent, index=self.adata_total.obs.index)
        self.latent_df.columns = ["latent_" + str(x) for x in range(1, self.n_latent + 1)]
        self.latent_df["batch"] = self.adata_total.obs["batch"]
        self.latent_df.to_csv(os.path.join(self.result_path, "latent.csv"))

        if self.ot:
            sc_celltype = self.latent_df[self.latent_df["batch"] == "scRNA-seq"]
            cluster_name = sc_celltype.unique()
            self.cluster_score = np.zeros((len(cluster_name), self.adata_B_input.shape[0]))
            self.cluster_score = pd.DataFrame(self.cluster_score, index=cluster_name,
                                              columns=self.adata_B_input.obs.index)
            for i in cluster_name:
                self.cluster_score.loc[i, :] = np.mean(self.plan[np.where(sc_celltype == i), :][0], axis=0)
            self.cluster_score = self.cluster_score.T
            self.cluster_score.to_csv(os.path.join(self.result_path, "cluster_score.csv"))

        if self.resolution == "high":
            self.trans_label = self.cluster_score.apply(lambda x: get_max_index(x), axis=1)
            self.trans_label = self.trans_label.replace(range(len(self.trans_label.unique())), self.cluster_score.columns)
            self.trans_label = pd.DataFrame(self.trans_label)
            self.trans_label.columns = ["transfer_label"]
            self.trans_label.index = self.adata_B_input.obs.index
            self.trans_label.to_csv(os.path.join(self.result_path, "trans_label.csv"))

            retain_cell = retain_prop * self.plan.shape[1]
            plan_filt = self.plan * (np.argsort(np.argsort(self.plan)) >= self.plan.shape[1] - retain_cell)
            self.plan_norm = plan_filt.T / np.sum(plan_filt, 1)
            self.plan_norm = sp.csr_matrix(self.plan_norm)
            self.data_pm = self.plan_norm @ self.adata_A_input.X
            self.adata_ST_pm = sc.AnnData(X=self.data_pm, obs=self.adata_B_input.obs, var=self.adata_A_input.var, obsm=self.adata_B_input.obsm)
            self.adata_ST_pm.write(os.path.join(self.result_path, "adata_ST_pm.h5ad"))
            if self.verbose:
                print("Enhanced ST dataset has been saved!")

        if D_score:
            self.D_A = discriminator(self.npcs).to(self.device)
            self.D_B = discriminator(self.npcs).to(self.device)
            self.D_A.load_state_dict(ckpt['D_A'])
            self.D_B.load_state_dict(ckpt['D_B'])
            score_D_A_A = self.D_A(x_A)
            score_D_B_A = self.D_B(x_AtoB)
            score_D_B_B = self.D_B(x_B)
            score_D_A_B = self.D_A(x_BtoA)
            self.score_Aspace = np.concatenate((score_D_A_A.detach().cpu().numpy(), score_D_A_B.detach().cpu().numpy()), axis=0)
            self.score_Bspace = np.concatenate((score_D_B_A.detach().cpu().numpy(), score_D_B_B.detach().cpu().numpy()), axis=0)

        if save_embedding:
            np.save(os.path.join(self.result_path, "latent_A.npy"), z_A.detach().cpu().numpy())
            np.save(os.path.join(self.result_path, "latent_B.npy"), z_B.detach().cpu().numpy())
            np.save(os.path.join(self.result_path, "x_AtoB.npy"), x_AtoB.detach().cpu().numpy())
            np.save(os.path.join(self.result_path, "x_BtoA.npy"), x_BtoA.detach().cpu().numpy())
            np.save(os.path.join(self.result_path, "map_A.npy"), m_A.detach().cpu().numpy())
            np.save(os.path.join(self.result_path, "map_B.npy"), m_B.detach().cpu().numpy())

        return mu, phi, sigma, z_A.detach().cpu().numpy(), z_B.detach().cpu().numpy(), m_A.detach().cpu().numpy(), m_B.detach().cpu().numpy()


# ---------- Conformal Prediction 函数 ----------
def conformal_prediction(true_coords, z_B, m_B, calib_index, test_index, alpha=0.05, k_neighbors=15):
    true_coords = torch.tensor(true_coords, dtype=torch.float)
    z_B = torch.tensor(z_B, dtype=torch.float)
    m_B = torch.tensor(m_B, dtype=torch.float)
    pred_coords = m_B.clone()
    lambda_test = torch.tensor(compute_lambda_test(true_coords.numpy(), z_B.numpy(), test_index, k_neighbors), dtype=torch.float).view(-1)
    lambda_calib = torch.tensor(compute_lambda_test(true_coords.numpy(), z_B.numpy(), calib_index, k_neighbors), dtype=torch.float).view(-1)
    v_test = torch.norm(true_coords[test_index] - pred_coords[test_index], dim=1) / lambda_test
    q = np.quantile(v_test.detach().numpy(), 1 - alpha)
    confidence_intervals = q * lambda_calib
    pred_errors = torch.norm(true_coords[calib_index, :] - pred_coords[calib_index], dim=1)
    aberrant = pred_errors > confidence_intervals

    # swap
    dist_matrix = torch.cdist(true_coords[calib_index], true_coords[calib_index])
    v_calib = torch.norm(true_coords[calib_index] - pred_coords[calib_index], dim=1) / lambda_calib
    q = np.quantile(v_calib.numpy(), 1 - alpha)
    confidence_intervals_test = q * lambda_test
    pred_errors_test = torch.norm(true_coords[test_index] - pred_coords[test_index], dim=1)
    aberrant_test = pred_errors_test > confidence_intervals_test

    n = true_coords.shape[0]
    final_aberrant = torch.zeros(n, dtype=torch.bool)
    final_aberrant[calib_index] = aberrant
    final_aberrant[test_index] = aberrant_test
    final_lambda = torch.zeros(n)
    final_conf = torch.zeros(n)
    final_lambda[calib_index] = lambda_calib.float()
    final_lambda[test_index] = lambda_test.float()
    final_conf[calib_index] = confidence_intervals.float()
    final_conf[test_index] = confidence_intervals_test.float()

    return final_aberrant.numpy(), final_conf.detach().numpy(), final_lambda, pred_coords, true_coords
