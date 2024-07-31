import time, os, random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
from torch.distributions.categorical import Categorical
import matplotlib as mpl
from scipy.stats import ortho_group, special_ortho_group
import copy
import torch.nn.functional as F


# import matlab.engine


# mpl.use('TkAgg')
# mpl.use('agg')


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def GMM_logprobs_Full(x, pi_q, mu_q, Sigma_q, epsilon=1e-8):
    '''
    K number of components, N number of samples, D dimension of each observation
    :param x:    N,D
    :param pi_q: K,1 or N,K
    :param mu_q: K,D
    :param std_q: K,D,D
    :return:
        log_gmm: N,1
        log_piNormal: N,K
        log_Normal: N,K
    '''
    N, D = x.shape
    eps = epsilon * torch.eye(D, dtype=mu_q.dtype)

    t1 = torch.log(pi_q).squeeze()  # K or N,K
    t2 = - D / 2. * torch.log(2. * pi)  #
    t3 = - 0.5 * torch.logdet(Sigma_q + eps)  # K
    x_mu = x.unsqueeze(1) - mu_q.unsqueeze(0)  # N,K,D
    t4 = - 0.5 * torch.matmul(
        x_mu.unsqueeze(-2),
        torch.matmul(torch.inverse(Sigma_q + eps), x_mu.unsqueeze(-1))
    ).squeeze(-1).squeeze(-1)  # N,K

    log_Normal = t2 + t3 + t4  # [N, K]
    log_piNormal = t1 + log_Normal  # [N, K]
    log_gmm = torch.logsumexp(log_piNormal, dim=1, keepdim=True)  # N,1

    return log_gmm, log_piNormal, log_Normal


def GMM_sample_n(pi_q, mu_q, std_q, n):
    dim_x = mu_q.shape[1]
    mode_var = Categorical(pi_q.squeeze())
    m = mode_var.sample((n,))
    data = mu_q[m, :] + std_q[m, :] * torch.randn([n, dim_x], device=device)
    return data.type(torch.float32), m


def GMM_sample_from_para(pi_p, para_mu_p, para_Sigma_p, n=1):
    """
    GMM sampling from trainable parameters, with gradient back propagation enabled for
        para_mu_p and para_Sigma_p
    :param pi_p: K,1
    :param para_mu_p: K,D
    :param para_Sigma_p: K, D(D+1)/2
    :param n: number of samples
    :return:
        samples: N,D
        z_samples: N
    """
    dim_x = para_mu_p.shape[1]
    mode_var = Categorical(pi_p.squeeze())
    z_samples = mode_var.sample((n,))

    Sigma_p, Lchol_p = para2Sigma(para_Sigma_p)  # K,D,D

    samples = para_mu_p[z_samples, :] + torch.matmul(
        Lchol_p[z_samples, :, :],
        torch.randn([n, dim_x, 1], device=device, dtype=para_mu_p.dtype)
    ).squeeze(-1)

    return samples, z_samples


def GMM_sample_from_Lchol(pi_p, mu_p, Lchol_p, n=1):
    """
    GMM sampling from trainable parameters, with gradient back propagation enabled for
        mu_p and Lchol_p
    :param pi_p: K,1
    :param mu_p: K,D
    :param Lchol_p: K,D,D
    :param n: number of samples
    :return:
        samples: N,D
        z_samples: N
    """
    dim_x = mu_p.shape[1]
    mode_var = Categorical(pi_p.detach().squeeze())
    z_samples = mode_var.sample((n,))

    samples = mu_p[z_samples, :] + torch.matmul(
        Lchol_p[z_samples, :, :],
        torch.randn([n, dim_x, 1], device=device, dtype=mu_p.dtype)
    ).squeeze(-1)

    return samples, z_samples


def GMM_sample_n_per_mode(mu_q, para_Sigma_q, n):
    # gradient BP is activated
    # mu_q: K,D
    # para_Sigma_q: K,L
    dim_x = mu_q.shape[1]
    _, Lchol = Lchol2Sigma(para_Sigma_q)  # K,D,D

    data = mu_q.unsqueeze(1) + torch.matmul(
        Lchol.unsqueeze(1),  # K,1,D,D
        torch.randn([n, dim_x, 1], device=device)
    ).squeeze(-1)  # K,n,D

    return data


def KL_gmm_Full(x_test, pi_q, mu_q, Sigma_q, pi_p, mu_p, Sigma_p):
    log_gmm_q, _, _ = GMM_logprobs_Full(x_test, pi_q, mu_q, Sigma_q)  # N,1
    log_gmm_p, _, _ = GMM_logprobs_Full(x_test, pi_p, mu_p, Sigma_p)  # N,1
    KL_test = (log_gmm_q - log_gmm_p).mean()
    return KL_test


def KL_gmm_condition_Full(x_test, pi_q, mu_q, Sigma_q, pi_p, mu_p, Sigma_p, Sindx, Tindx):
    log_gmm_qST, _, _ = GMM_logprobs_Full(x_test[:, Sindx + Tindx], pi_q,
                                          mu_q[:, Sindx + Tindx],
                                          Sigma_q[:, Sindx + Tindx, :][:, :, Sindx + Tindx])  # N,1  N,K
    log_gmm_qS, _, _ = GMM_logprobs_Full(x_test[:, Sindx], pi_q,
                                         mu_q[:, Sindx],
                                         Sigma_q[:, Sindx, :][:, :, Sindx])  # N,1  N,K
    log_gmm_qTgS = log_gmm_qST - log_gmm_qS

    log_gmm_pST, _, _ = GMM_logprobs_Full(x_test[:, Sindx + Tindx], pi_p,
                                          mu_p[:, Sindx + Tindx],
                                          Sigma_p[:, Sindx + Tindx, :][:, :, Sindx + Tindx])  # N,1  N,K
    log_gmm_pS, _, _ = GMM_logprobs_Full(x_test[:, Sindx], pi_p,
                                         mu_p[:, Sindx],
                                         Sigma_p[:, Sindx, :][:, :, Sindx])  # N,1  N,K
    log_gmm_pTgS = log_gmm_pST - log_gmm_pS

    KL_test = (log_gmm_qTgS - log_gmm_pTgS).mean()
    return KL_test


def contourplot_gmm_Full(pi_q, mu_q, Sigma_q, pi_p, mu_p, Sigma_p, x_all=None, num_points=100, fontsize=14):
    inds = torch.linspace(-2.5, 2.5, num_points)
    inds_y, inds_x = torch.meshgrid(inds, inds)
    XY = torch.cat((inds_x.reshape(-1, 1), inds_y.reshape(-1, 1)), dim=1)  # N,D

    QXY, _, _ = GMM_logprobs_Full(XY, pi_q, mu_q, Sigma_q)  # N,1
    PXY, _, _ = GMM_logprobs_Full(XY, pi_p, mu_p, Sigma_p)  # N,1
    QXY = QXY.exp().reshape(num_points, num_points)
    PXY = PXY.exp().reshape(num_points, num_points)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    cntr1 = ax.contour(inds_x.numpy(), inds_y.numpy(), PXY.numpy(), 8, alpha=0.5,
                       colors='green')
    cntr2 = ax.contour(inds_x.numpy(), inds_y.numpy(), QXY.numpy(), 8, alpha=0.5,
                       colors='red')
    h1, _ = cntr1.legend_elements()
    h2, _ = cntr2.legend_elements()
    ax.legend([h1[0], h2[0]], [r'$p_{\theta}(x)$', r'$q(x)$'], fontsize=fontsize + 2)
    for ii in range(pi_p.numel()):
        ax.annotate(r'$\pi_{%d}=%.2f$' % (ii, pi_p[ii].numpy()),
                    xy=(mu_p[ii, 0].numpy(), mu_p[ii, 1].numpy()),
                    xytext=(+15, -15), textcoords='offset points', fontsize=fontsize,
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

    if x_all is not None:
        plt.scatter(x_all[:, 0], x_all[:, 1])


def contourplot_gmm_Full_1(pi_q, mu_q, Sigma_q, pi_p, mu_p, Sigma_p, x_all=None, num_points=100, fontsize=14):
    inds = torch.linspace(-5.0, 5.0, num_points)
    inds_y, inds_x = torch.meshgrid(inds, inds)
    XY = torch.cat((inds_x.reshape(-1, 1), inds_y.reshape(-1, 1)), dim=1)  # N,D

    QXY, _, _ = GMM_logprobs_Full(XY, pi_q, mu_q, Sigma_q)  # N,1
    PXY, _, _ = GMM_logprobs_Full(XY, pi_p, mu_p, Sigma_p)  # N,1
    QXY = QXY.exp().reshape(num_points, num_points)
    PXY = PXY.exp().reshape(num_points, num_points)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    cntr1 = ax.contour(inds_x.numpy(), inds_y.numpy(), PXY.numpy(), 8, alpha=0.5,
                       colors='green')
    cntr2 = ax.contour(inds_x.numpy(), inds_y.numpy(), QXY.numpy(), 8, alpha=0.5,
                       colors='red')
    h1, _ = cntr1.legend_elements()
    h2, _ = cntr2.legend_elements()
    ax.legend([h1[0], h2[0]], [r'$p_{\theta}(x)$', r'$q(x)$'], fontsize=fontsize + 2)
    # for ii in range(pi_p.numel()):
    #     ax.annotate(r'$\pi_{%d}$' % (ii),
    #                 xy=(mu_p[ii, 0].numpy(), mu_p[ii, 1].numpy()),
    #                 xytext=(+20, -20), textcoords='offset points', fontsize=fontsize,
    #                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

    if x_all is not None:
        plt.scatter(x_all[:, 0], x_all[:, 1])


def contourplot_gmm(pi_q, mu_q, Sigma_q, num_points=100):
    inds = torch.linspace(-2, 2, num_points)
    inds_y, inds_x = torch.meshgrid(inds, inds)
    XY = torch.cat((inds_x.reshape(-1, 1), inds_y.reshape(-1, 1)), dim=1)  # N,D

    QXY, _, _ = GMM_logprobs_Full(XY, pi_q, mu_q, Sigma_q)  # N,1
    QXY = QXY.exp().reshape(num_points, num_points)

    fig, ax = plt.subplots(dpi=100)
    cntr1 = ax.contour(inds_x.numpy(), inds_y.numpy(), QXY.numpy(), 18, alpha=0.5,
                       colors='green')
    h1, _ = cntr1.legend_elements()


def rand_mat(dim=2, method='orthogonal'):
    # method = 'rotation', 'orthogonal'
    if method == 'orthogonal':
        A = torch.tensor(ortho_group.rvs(dim), dtype=torch.float32)  # Orthogonal
    elif method == 'rotation':
        A = torch.tensor(special_ortho_group.rvs(dim), dtype=torch.float32)  # Rotation
    else:
        print('Invalid input for <method>')
    return A


def correct_Sigma(Sigma_p, eps=1e-5):  # , method='max', eps=1e-2):
    if True:
        L, V = torch.linalg.eigh(Sigma_p)
        # torch.dist(V @ torch.diag_embed(L) @ torch.linalg.inv(V), Sigma_p)
        Sigma_p = V @ torch.diag_embed(
            torch.maximum(torch.tensor(eps), L)
        ) @ torch.linalg.inv(V)
    else:
        _, info = torch.linalg.cholesky_ex(Sigma_p)
        if (info > 0).any():
            L, V = torch.linalg.eigh(Sigma_p)
            # torch.dist(V @ torch.diag_embed(L) @ torch.linalg.inv(V), Sigma_p)
            Sigma_p = V @ torch.diag_embed(
                torch.maximum(torch.tensor(eps), L)
            ) @ torch.linalg.inv(V)
            # K, D, _ = Sigma_p.shape
            # Ks, Ds = torch.meshgrid(torch.arange(K), torch.arange(D))
            # if method == 'max':
            #     L[Ks, Ds, Ds] = torch.maximum(torch.tensor(eps), L[Ks, Ds, Ds])
            # else:
            #     L[Ks, Ds, Ds] = L[Ks, Ds, Ds].abs()
            # Sigma_p = torch.matmul(L, torch.transpose(L, 1, 2))
            # Sigma_p1 = torch.matmul(L, torch.transpose(L, 1, 2))
            # L1, info1 = torch.linalg.cholesky_ex(Sigma_p1)
    return Sigma_p


def Sigma2Lchol(Sigma_p):
    # GradientBP is not necessary
    L, info = torch.linalg.cholesky_ex(Sigma_p)

    D = Sigma_p.shape[1]  # K,D,D
    L[:, torch.arange(D), torch.arange(D)] = L[:, torch.arange(D), torch.arange(D)].log()

    trilindx = torch.tril_indices(D, D)
    para_Sigma_p = L[:, trilindx[0], trilindx[1]]
    return para_Sigma_p


def para2Sigma(para_Sigma_p):
    """
    tranform the para_Sigma_p into Sigma_p, with gradient back propagation enabled
    :param para_Sigma_p: K, D(D+1)/2
    :return:
        Sigma_p: K,D,D
        Lchol_p: K,D,D. Lower triangle matrix of chol_decomp
    """
    K, DD2 = para_Sigma_p.shape
    D = int((np.sqrt(1 + 8 * DD2) - 1) / 2)
    Lchol_p = torch.zeros(K, D, D, dtype=para_Sigma_p.dtype)

    trilindx = torch.tril_indices(D, D)
    Lchol_p[:, trilindx[0], trilindx[1]] = para_Sigma_p
    Lchol_p[:, torch.arange(D), torch.arange(D)] = Lchol_p[:, torch.arange(D), torch.arange(D)].exp()  # K,D,D

    Sigma_p = torch.matmul(Lchol_p, torch.permute(Lchol_p, (0, 2, 1)))

    return Sigma_p, Lchol_p


def GMM_logprobs_Full_pl(x, pi_q, mu_q, Sigma_q):
    '''
    K number of components, N number of samples, D dimension of each observation
    :param x:    M1,M2,N,D
    :param pi_q: M1,M2,K,1  or 1,1,K,1
    :param mu_q: M1,M2,K,D  or 1,1,K,D
    :param std_q: M1,M2,K,D,D  or 1,1,K,D,D
    :return:
    '''
    M1, M2, N, D = x.shape

    t1 = torch.log(pi_q).squeeze().unsqueeze(-2)  # M1,M2,1,K  or 1,1,1,K
    t2 = - D / 2. * torch.log(2. * pi)  # 1
    t3 = - 0.5 * torch.logdet(Sigma_q).unsqueeze(-2)  # M1,M2,1,K  or 1,1,1,K
    x_mu = x.unsqueeze(-2) - mu_q.unsqueeze(-3)  # M1,M2,N,K,D
    t4 = - 0.5 * torch.matmul(
        x_mu.unsqueeze(-2),
        torch.matmul(torch.inverse(Sigma_q).unsqueeze(-4), x_mu.unsqueeze(-1))  # M1,M2,N,K,D,1
    ).squeeze(-1).squeeze(-1)  # M1,M2,N,K

    log_Normal = t2 + t3 + t4  # M1,M2,N,K
    log_piNormal = t1 + log_Normal  # M1,M2,N,K
    log_gmm = torch.logsumexp(log_piNormal, dim=-1, keepdim=False)  # M1,M2,N

    return log_gmm, log_piNormal, log_Normal


def GMM_sample_n_paraLchol_pl(pi_q, mu_q, Lchol_q, n):
    '''
    K number of components, N number of samples, D dimension of each observation
        :param pi_q: M1,M2,K,1  or 1,1,K,1
        :param mu_q: M1,M2,K,D  or 1,1,K,D
        :param Lchol_q: M1,M2,K,D,D  or 1,1,K,D,D
    return:
        :param x:    M1,M2,N,D
    '''
    M1, M2, K, dim_x = mu_q.shape
    mode_var = Categorical(pi_q.squeeze(-1))
    m = mode_var.sample((n,)).permute(1, 2, 0)  # M1,M2,N  or 1,1,N

    if m.shape[0] < M1:
        m = m.repeat(M1, M2, 1)
    if Lchol_q.shape[0] < M1:
        Lchol_q = Lchol_q.repeat(M1, M2, 1, 1, 1)

    data = torch.gather(mu_q, dim=-2, index=m.unsqueeze(-1).repeat(1, 1, 1, dim_x)) + torch.matmul(
        torch.gather(Lchol_q, dim=-3, index=m.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, dim_x, dim_x)),
        torch.randn([M1, M2, n, dim_x, 1], device=device)
    ).squeeze(-1)  # M1,M2,N,D

    # data = torch.gather(mu_q, dim=-2, index=m.repeat(M1, M2, 1).unsqueeze(-1).repeat(1, 1, 1, dim_x)) + torch.matmul(
    #     torch.gather(Lchol_q, dim=-3, index=m.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, dim_x, dim_x)),
    #     torch.randn([M1, M2, n, dim_x, 1], device=device)
    # ).squeeze(-1)

    return data.type(torch.float32), m


def Lchol2Sigma_pl(para_Sigma_p):
    '''
    # GradientBP is necessary
    para_Sigma_p: M1,M2,K,L  or 1,1,K,L
    Sigma_p: M1,M2,K,D,D  or 1,1,K,D,D
    '''
    # K, D = para_Sigma_p.shape
    M1, M2, K, L = para_Sigma_p.shape
    D = int((np.sqrt(8 * L) - 1) / 2)
    Lchol = torch.zeros(M1, M2, K, D, D, dtype=para_Sigma_p.dtype)

    trilindx = torch.tril_indices(x_dim, x_dim)
    Lchol[:, :, :, trilindx[0], trilindx[1]] = para_Sigma_p
    Lchol[:, :, :, torch.arange(D), torch.arange(D)] = L[:, :, :, torch.arange(D), torch.arange(D)].exp()  # K,D,D

    Sigma_p = torch.matmul(L, torch.permute(L, (-3, -1, -2)))
    return Sigma_p, Lchol


def Sigma2Lchol_pl(Sigma_p):
    '''
    # GradientBP is not necessary
    Sigma_p: M1,M2,K,D,D  or 1,1,K,D,D
    para_Sigma_p: M1,M2,K,L  or 1,1,K,L
    '''
    Lchol, info = torch.linalg.cholesky_ex(Sigma_p)
    TMP = copy.deepcopy(Lchol)

    M1, M2, K, D, _ = Sigma_p.shape  # M1,M2,K,D,D
    TMP[:, :, :, torch.arange(D), torch.arange(D)] = TMP[:, :, :, torch.arange(D), torch.arange(D)].log()

    trilindx = torch.tril_indices(D, D)
    para_Sigma_p = TMP[:, :, :, trilindx[0], trilindx[1]]
    return para_Sigma_p, Lchol


def rand_mat(dim=2, method='orthogonal'):
    # method = 'rotation', 'orthogonal'
    if method == 'orthogonal':
        A = torch.tensor(ortho_group.rvs(dim), dtype=torch.float32)  # Orthogonal
    elif method == 'rotation':
        A = torch.tensor(special_ortho_group.rvs(dim), dtype=torch.float32)  # Rotation
    else:
        print('Invalid input for <method>')
    return A


## Settings
# eng = matlab.engine.start_matlab()
pi = torch.tensor(math.pi)
epsilon = torch.tensor(1e-5)
device = 'cpu'

random_seed = 888  # 888     5079, 6395, 3325, 2580, 5755, 2488, 4515, 6607, 2432, 6064
seed_torch(random_seed)

out_dir = 'E:/ExperiResults/BigLearnRKL/GMM25_%d/' % (random_seed)
os.makedirs(out_dir) if not os.path.isdir(out_dir) else None

# settings: data
x_dim = 2
var_q = 0.05
num_component = 25

# settings: train
Niter = 10000
NITnei = 20
lrSGD = 0.1
num_sample_p = 100
Amethod = 'orthogonal'  # 'orthogonal', 'rotation'
plotIT = 200

# -----   data   -------
pi_q = 1. / num_component * torch.ones(num_component, 1, device=device)
mu_q = torch.tensor([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5],
                     [2, 1], [2, 2], [2, 3], [2, 4], [2, 5],
                     [3, 1], [3, 2], [3, 3], [3, 4], [3, 5],
                     [4, 1], [4, 2], [4, 3], [4, 4], [4, 5],
                     [5, 1], [5, 2], [5, 3], [5, 4], [5, 5]], device=device, dtype=torch.float32) * 2 - 6
Sigma_q = torch.diag_embed(var_q * torch.ones(num_component, x_dim, device=device))  # K,D,D

# -----   model   -------
para_pi_p = torch.ones(num_component, 1, device=device, requires_grad=True)
para_mu_p = -5 + 0.1 * torch.randn(num_component, x_dim, device=device)
para_Sigma_p = 0. * torch.ones(num_component, int(x_dim * (x_dim + 1) / 2), device=device)
para_mu_p.requires_grad = True
para_Sigma_p.requires_grad = True

optimizer = torch.optim.SGD([para_mu_p, para_Sigma_p], lr=lrSGD)

with torch.no_grad():
    pi_p = torch.softmax(para_pi_p, dim=0)  # K,1
    mu_p = para_mu_p
    Sigma_p, Lchol_p = para2Sigma(para_Sigma_p)

    contourplot_gmm_Full_1(pi_q, mu_q, Sigma_q, pi_p, mu_p, Sigma_p, fontsize=32)
    plt.xlabel('$x_1$', fontsize=36)
    plt.ylabel('$x_2$', fontsize=36)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.savefig(out_dir + '0000_BigLearnRKL.png', dpi=300, bbox_inches='tight')
    plt.close()

# -----   Train   -------
JMCprobs = [0., 1.]
for IT in range(Niter):
    '''
    J M C
    if M, sample A, do y1 marginal matching
    if C, sample A, sample y2, do y1|y2 conditional matching
    '''
    if IT == int(Niter / 2):
        JMCprobs = [0.1, 1.]

    urand = torch.rand(1)
    if urand <= JMCprobs[0]:
        do = 'joint_matching'
    elif urand <= JMCprobs[1]:
        do = 'marginal_matching'
        A = rand_mat(dim=x_dim, method=Amethod)
        bar_mu_q = mu_q.mm(A.t())
        bar_Sigma_q = torch.matmul(A.unsqueeze(0), torch.matmul(Sigma_q, A.t()))
    else:
        do = 'conditional_matching'
        A = rand_mat(dim=x_dim, method=Amethod)
        bar_mu_q = mu_q.mm(A.t())
        bar_Sigma_q = torch.matmul(A.unsqueeze(0), torch.matmul(Sigma_q, A.t()))

    for iii in range(NITnei):

        #  ------------     sample from p(x)    ------------
        pi_p = torch.softmax(para_pi_p, dim=0)  # K,1
        mu_p = para_mu_p
        Sigma_p, Lchol_p = para2Sigma(para_Sigma_p)

        #  ------------     do == 'joint_matching'    ------------
        if do == 'joint_matching':
            samples, z_samples = GMM_sample_from_Lchol(pi_p.detach(), mu_p, Lchol_p, n=num_sample_p)  # N,D   N

            log_gmm_p, log_piNormal_p, log_Normal_p = GMM_logprobs_Full(samples, pi_p.detach(),
                                                                        mu_p.detach(), Sigma_p.detach())
            log_gmm_q, log_piNormal_q, log_Normal_q = GMM_logprobs_Full(samples, pi_q, mu_q, Sigma_q)

            z_proj = F.one_hot(z_samples, num_classes=num_component).t()  # K,N
            z_proj = z_proj / torch.maximum(epsilon, z_proj.sum(1, keepdim=True))  # K,N

            loss = (pi_p * z_proj.mm(log_gmm_p - log_gmm_q)).sum()

            # print('ITnei=%d, loss=%.3f' % (iii, loss.item()))

        # ------------    do == 'marginal_matching'    ------------
        elif do == 'marginal_matching':
            samples, z_samples = GMM_sample_from_Lchol(pi_p.detach(), mu_p, Lchol_p, n=num_sample_p)  # N,D   N

            bar_mu_p = mu_p.mm(A.t())
            bar_Sigma_p = torch.matmul(A.unsqueeze(0), torch.matmul(Sigma_p, A.t()))
            bar_samples = samples.mm(A.t())

            log_gmm1_p, _, _ = GMM_logprobs_Full(bar_samples[:, 0:1], pi_p.detach(),
                                                 bar_mu_p[:, 0:1].detach(), bar_Sigma_p[:, 0:1, 0:1].detach())
            log_gmm1_q, _, _ = GMM_logprobs_Full(bar_samples[:, 0:1], pi_q,
                                                 bar_mu_q[:, 0:1], bar_Sigma_q[:, 0:1, 0:1])

            z_proj = F.one_hot(z_samples, num_classes=num_component).t()  # K,N
            z_proj = z_proj / torch.maximum(epsilon, z_proj.sum(1, keepdim=True))  # K,N

            loss = (pi_p * z_proj.mm(log_gmm1_p - log_gmm1_q)).sum()

            # print('ITnei=%d, loss=%.3f' % (iii, loss.item()))

        # ------------     do == 'conditional_matching'    ------------
        else:
            bar_mu_p = mu_p.mm(A.t())
            bar_Sigma_p = torch.matmul(A.unsqueeze(0), torch.matmul(Sigma_p, A.t()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #   --------------------     Figure     ----------------------
    if IT % plotIT == 0:
        print('IT=%d, loss=%.3f' % (IT, loss.item()))
        with torch.no_grad():
            pi_p = torch.softmax(para_pi_p, dim=0)  # K,1
            mu_p = para_mu_p
            Sigma_p, Lchol_p = para2Sigma(para_Sigma_p)

            contourplot_gmm_Full_1(pi_q, mu_q, Sigma_q, pi_p, mu_p, Sigma_p, fontsize=32)
            plt.xlabel('$x_1$', fontsize=36)
            plt.ylabel('$x_2$', fontsize=36)
            plt.xticks(fontsize=32)
            plt.yticks(fontsize=32)
            plt.savefig(out_dir + '%d_BigLearnRKL.png' % IT, dpi=300, bbox_inches='tight')
            plt.close()

curr_time = time.strftime("%H:%M:%S", time.localtime())
print("Current Time is :", curr_time)

# eng.quit()
