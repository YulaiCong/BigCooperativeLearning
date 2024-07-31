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


def GMM_logprobs_Full(x, pi_q, mu_q, Sigma_q, mask=None):
    '''
    K number of components, N number of samples, D dimension of each observation
    :param x:    N,D
    :param pi_q: K,1 or N,K
    :param mu_q: K,D
    :param std_q: K,D,D
    :param mask: N,D
    :return:
    '''
    N, D = x.shape

    t1 = torch.log(pi_q).squeeze()  # K or N,K
    t2 = - D / 2. * torch.log(2. * pi)  #
    t3 = - 0.5 * torch.logdet(Sigma_q)  # K
    x_mu = x.unsqueeze(1) - mu_q.unsqueeze(0)  # N,K,D
    t4 = - 0.5 * torch.matmul(
        x_mu.unsqueeze(-2),
        torch.matmul(torch.inverse(Sigma_q), x_mu.unsqueeze(-1))
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


def GMM_sample_n_paraLchol(pi_q, mu_q, para_Sigma_q, n):
    dim_x = mu_q.shape[1]
    mode_var = Categorical(pi_q.squeeze())
    m = mode_var.sample((n,))

    _, Lchol = Lchol2Sigma(para_Sigma_q)  # K,D,D

    data = mu_q[m, :] + torch.matmul(
        Lchol[m, :, :],
        torch.randn([n, dim_x, 1], device=device)
    ).squeeze(-1)

    return data.type(torch.float32), m


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


def Lchol2Sigma(para_Sigma_p):
    # GradientBP is necessary
    # K, D = para_Sigma_p.shape
    K, _ = para_Sigma_p.shape
    D = x_dim
    L = torch.zeros(K, D, D, dtype=para_Sigma_p.dtype)

    trilindx = torch.tril_indices(x_dim, x_dim)
    L[:, trilindx[0], trilindx[1]] = para_Sigma_p
    L[:, torch.arange(D), torch.arange(D)] = L[:, torch.arange(D), torch.arange(D)].exp()  # K,D,D

    Sigma_p = torch.matmul(L, torch.permute(L, (0, 2, 1)))
    return Sigma_p, L


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
epsilonPi = torch.tensor(0.01)
device = 'cpu'

random_seed = 888  # 999
do_savefig = True
do_pltshow = False

Niter = 3000
plotIT = 50
lrSGD = 1e-2  # 5e-3
n_permode = 10

# out_dir = 'E:/ExperiResults/BigLearn/'
out_dir = './Figure_RKL/'
os.makedirs(out_dir) if not os.path.isdir(out_dir) else None

seed_torch(random_seed)

x_dim = 2
num_data = 2000
num_data_test = 5000

num_component = 2
sigma2 = 0.1  # 0.2  0.1   0.05   0.02
gamma_color = 0.5

pi_q = 1. / num_component * torch.ones(num_component, 1, device=device)
mu_q = torch.tensor([[-1, 0], [1, 0]], device=device, dtype=torch.float32)
std_q = torch.sqrt(torch.tensor([sigma2], device=device)) * torch.ones(num_component, x_dim, device=device)
Sigma_q = torch.diag_embed(std_q.pow(2))  # K,D,D

pi_q_pl = pi_q.unsqueeze(-3).unsqueeze(-4)  # 1,1,K,1
mu_q_pl = mu_q.unsqueeze(-3).unsqueeze(-4)  # 1,1,K,L
Sigma_q_pl = Sigma_q.unsqueeze(-4).unsqueeze(-5)  # 1,1,K,D,D
# para_Sigma_q_pl = Sigma2Lchol(Sigma_q).unsqueeze(-3).unsqueeze(-4)  # 1,1,K,L

x_train, y_train = GMM_sample_n(pi_q, mu_q, std_q, num_data)
x_test, y_test = GMM_sample_n(pi_q, mu_q, std_q, num_data_test)

# plt.scatter(x_train[:, 0], x_train[:, 1], s=5, c=y_train)
# plt.xlim(-2.5, 2.5), plt.ylim(-1.5, 1.5)
# plt.title('training data')
# plt.show()

plt.scatter(x_test[:, 0], x_test[:, 1], s=5, c=y_test)
plt.xlim(-2.5, 2.5), plt.ylim(-1.5, 1.5)
plt.title('testing data')
plt.show()

contourplot_gmm(pi_q, mu_q, Sigma_q)
plt.xlabel('$x_1$', fontsize=20), plt.ylabel('$x_2$', fontsize=20)
plt.axis('square')
plt.axis([-2, 2, -2, 2])
plt.gcf().subplots_adjust(bottom=0.15)
if do_savefig:
    plt.savefig(out_dir + 'qx.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(out_dir + 'qx.png', dpi=300, bbox_inches='tight')
if do_pltshow:
    plt.show()
else:
    plt.close()

pi_p = 1. / num_component * torch.ones(num_component, 1, device=device)
mu_p = torch.randn(num_component, x_dim, device=device) + torch.tensor([[-2, 0]], device=device)
Sigma_p = torch.diag_embed(torch.ones(num_component, x_dim, device=device)) * sigma2  # 0.1

## --------------------   Joint Reverse KL loss    -------------------------------
n = 2000
steps = 100

pi_p_pl = 1. / num_component * torch.ones(1, 1, num_component, 1, device=device)  # 1,1,K,1
Sigma_p_pl = torch.diag_embed(torch.ones(1, 1, num_component, x_dim, device=device)) * sigma2  # 1,1,K,D,D
para_Sigma_p_pl, Lchol_p_pl = Sigma2Lchol_pl(Sigma_p_pl)  # 1,1,K,L
nums = torch.linspace(-2, 2, steps, device=device)
grid_mu1, grid_mu2 = torch.meshgrid(nums, nums, indexing='ij')
mu_p_pl = torch.zeros(steps, steps, num_component, x_dim, device=device)  # M1,M2,K,D
mu_p_pl[:, :, :, 0] = torch.stack((grid_mu1, grid_mu2), dim=-1)

if True:
    print('Processing Joint Reverse KL ...')
    seed_torch(random_seed)

    with torch.no_grad():
        samples_p, m = GMM_sample_n_paraLchol_pl(pi_p_pl, mu_p_pl, Lchol_p_pl, n)

        log_gmm_p, _, _ = GMM_logprobs_Full_pl(samples_p, pi_p_pl, mu_p_pl, Sigma_p_pl)  # M1,M2,N
        log_gmm_q, _, _ = GMM_logprobs_Full_pl(samples_p, pi_q_pl, mu_q_pl, Sigma_q_pl)  # M1,M2,N

        KL_test = (log_gmm_p - log_gmm_q).mean(-1)

    max_colorbar = KL_test.abs().reshape(-1).max()

    ax = plt.figure().add_subplot(projection='3d')
    # ax.plot_surface(grid_mu1, grid_mu2, KL_test, edgecolor='royalblue', lw=0.1, rstride=8, cstride=8,
    #                 alpha=0.3)
    # ax.plot_surface(grid_mu1.cpu(), grid_mu2.cpu(), KL_test.cpu(), cmap=plt.cm.YlGnBu_r)
    ax.plot_surface(grid_mu1.cpu(), grid_mu2.cpu(), KL_test.abs().cpu(), cmap='coolwarm')
    # ax.contour(grid_mu1, grid_mu2, KL_test, zdir='z', offset=-100, cmap='coolwarm')
    # ax.contour(grid_mu1, grid_mu2, KL_test, zdir='x', offset=-40, cmap='coolwarm')
    # ax.contour(grid_mu1, grid_mu2, KL_test, zdir='y', offset=40, cmap='coolwarm')
    ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(0, KL_test.abs().cpu().max()),
           xlabel='$\mu_1$', ylabel='$\mu_2$', zlabel='Joint Reverse KL')
    # ax.scatter(1, -1, 0, color='r', marker="*")
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.zaxis.label.set_size(16)
    # plt.plot(-1, 1, 0, 'r', marker="*", markersize=20, zorder=5)
    # plt.plot(1, -1, 0, 'r', marker="*", markersize=20, zorder=5)
    if do_savefig:
        # plt.gcf().subplots_adjust(left=-0.2, right=1.2, top=1., bottom=0.03)
        plt.savefig(out_dir + 'JointRKL_mesh.pdf', dpi=300)
        plt.savefig(out_dir + 'JointRKL_mesh.png', dpi=300)
    if do_pltshow:
        plt.show()
    else:
        plt.close()

    plt.pcolormesh(grid_mu1.cpu(), grid_mu2.cpu(), KL_test.abs().cpu(),
                   norm=colors.PowerNorm(gamma=gamma_color), cmap='coolwarm')
    plt.plot(-1, 1, 'r', marker="*", markersize=20, zorder=5)
    plt.plot(1, -1, 'r', marker="*", markersize=20, zorder=5)
    plt.xlabel('$\mu_1$', fontsize=20), plt.ylabel('$\mu_2$', fontsize=20)
    plt.axis('square')
    cbar = plt.colorbar()
    cbar.mappable.set_clim(0, max_colorbar)
    # plt.gcf().subplots_adjust(bottom=0.12)
    if do_savefig:
        plt.savefig(out_dir + 'JointRKL_meshxy.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(out_dir + 'JointRKL_meshxy.png', dpi=300, bbox_inches='tight')
    if do_pltshow:
        plt.show()
    else:
        plt.close()

    # eng.mesh(np.ascontiguousarray(grid_mu1),
    #          np.ascontiguousarray(grid_mu2),
    #          np.ascontiguousarray(KL_test))

## --------------------   Margin reverse KL loss    -------------------------------
if False:
    print('Processing Margin Reverse KL ...')
    seed_torch(random_seed)

    with torch.no_grad():
        samples_p, m = GMM_sample_n_paraLchol_pl(pi_p_pl, mu_p_pl, Lchol_p_pl, n)

        log_gmm_p, _, _ = GMM_logprobs_Full_pl(samples_p[:, :, :, :1], pi_p_pl, mu_p_pl[:, :, :, :1],
                                               Sigma_p_pl[:, :, :, :1, :1])  # M1,M2,N
        log_gmm_q, _, _ = GMM_logprobs_Full_pl(samples_p[:, :, :, :1], pi_q_pl, mu_q_pl[:, :, :, :1],
                                               Sigma_q_pl[:, :, :, :1, :1])  # M1,M2,N

        KL_test_m1 = (log_gmm_p - log_gmm_q).mean(-1)

        # samples_p_m2, mm2 = GMM_sample_n_paraLchol_pl(pi_p_pl, mu_p_pl[:,:,:,1:], Lchol_p_pl[:,:,:,1:,1:], n)
        # log_gmm_p, _, _ = GMM_logprobs_Full_pl(samples_p_m2, pi_p_pl, mu_p_pl[:,:,:,1:], Sigma_p_pl[:,:,:,1:,1:])  # M1,M2,N
        # log_gmm_q, _, _ = GMM_logprobs_Full_pl(samples_p_m2, pi_q_pl, mu_q_pl[:,:,:,1:], Sigma_q_pl[:,:,:,1:,1:])  # M1,M2,N

        log_gmm_p, _, _ = GMM_logprobs_Full_pl(samples_p[:, :, :, 1:], pi_p_pl, mu_p_pl[:, :, :, 1:],
                                               Sigma_p_pl[:, :, :, 1:, 1:])  # M1,M2,N
        log_gmm_q, _, _ = GMM_logprobs_Full_pl(samples_p[:, :, :, 1:], pi_q_pl, mu_q_pl[:, :, :, 1:],
                                               Sigma_q_pl[:, :, :, 1:, 1:])  # M1,M2,N

        KL_test_m2 = (log_gmm_p - log_gmm_q).mean(-1)  # ZERO

    ax = plt.figure().add_subplot(projection='3d')
    # ax.plot_surface(grid_mu1.cpu(), grid_mu2.cpu(), KL_test_m1.cpu(), cmap=plt.cm.YlGnBu_r)
    ax.plot_surface(grid_mu1.cpu(), grid_mu2.cpu(), KL_test_m1.cpu(), cmap='coolwarm')
    ax.set(xlim=(-2, 2), ylim=(-2, 2),  # zlim=(0, KL_test_m1.max()),
           xlabel='$\mu_1$', ylabel='$\mu_2$', zlabel='Margin Reverse KL')
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.zaxis.label.set_size(16)
    if do_savefig:
        plt.savefig(out_dir + 'MarginRKL_mesh1.pdf', dpi=300)
        plt.savefig(out_dir + 'MarginRKL_mesh1.png', dpi=300)
    if do_pltshow:
        plt.show()
    else:
        plt.close()

    plt.pcolormesh(grid_mu1.cpu(), grid_mu2.cpu(), KL_test_m1.abs().cpu(),
                   norm=colors.PowerNorm(gamma=gamma_color), cmap='coolwarm')
    plt.plot(-1, 1, 'r', marker="*", markersize=20, zorder=5)
    plt.plot(1, -1, 'r', marker="*", markersize=20, zorder=5)
    plt.xlabel('$\mu_1$', fontsize=20), plt.ylabel('$\mu_2$', fontsize=20)
    plt.axis('square')
    cbar = plt.colorbar()
    cbar.mappable.set_clim(0, max_colorbar)
    cbar.remove()
    if do_savefig:
        plt.savefig(out_dir + 'MarginRKL_meshxy1.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(out_dir + 'MarginRKL_meshxy1.png', dpi=300, bbox_inches='tight')
    if do_pltshow:
        plt.show()
    else:
        plt.close()

    ax = plt.figure().add_subplot(projection='3d')
    # ax.plot_surface(grid_mu1.cpu(), grid_mu2.cpu(), KL_test_m2.cpu(), cmap=plt.cm.YlGnBu_r)
    ax.plot_surface(grid_mu1.cpu(), grid_mu2.cpu(), KL_test_m2.cpu(), cmap='coolwarm')
    ax.set(xlim=(-2, 2), ylim=(-2, 2),  # zlim=(0, KL_test_m1.max()),
           xlabel='$\mu_1$', ylabel='$\mu_2$', zlabel='Margin Reverse KL')
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.zaxis.label.set_size(16)
    if do_savefig:
        plt.savefig(out_dir + 'MarginRKL_mesh2.pdf', dpi=300)
        plt.savefig(out_dir + 'MarginRKL_mesh2.png', dpi=300)
    if do_pltshow:
        plt.show()
    else:
        plt.close()

    plt.pcolormesh(grid_mu1.cpu(), grid_mu2.cpu(), KL_test_m2.abs().cpu(),
                   norm=colors.PowerNorm(gamma=gamma_color), cmap='coolwarm')
    plt.plot(-1, 1, 'r', marker="*", markersize=20, zorder=5)
    plt.plot(1, -1, 'r', marker="*", markersize=20, zorder=5)
    plt.xlabel('$\mu_1$', fontsize=20), plt.ylabel('$\mu_2$', fontsize=20)
    plt.axis('square')
    cbar = plt.colorbar()
    cbar.mappable.set_clim(0, max_colorbar)
    cbar.remove()
    if do_savefig:
        plt.savefig(out_dir + 'MarginRKL_meshxy2.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(out_dir + 'MarginRKL_meshxy2.png', dpi=300, bbox_inches='tight')
    if do_pltshow:
        plt.show()
    else:
        plt.close()

## --------------------   Transformed Margin Reverse KL loss    -------------------------------
if True:
    print('Processing Transformed Margin Reverse KL ...')

    seed_torch(random_seed)

    rotate_angles = np.linspace(0, 180, 13)

    x_all, y_all = GMM_sample_n(pi_q, mu_q, std_q, num_data)

    bar_KL_test = 0
    with torch.no_grad():
        samples_p, m = GMM_sample_n_paraLchol_pl(pi_p_pl, mu_p_pl, Lchol_p_pl, n)  # M1,M2,N,D

        # pi_p_pl   # 1,1,K,1
        # mu_p_pl   # M1,M2,K,D
        # Sigma_p_pl  # 1,1,K,D,D
        # Lchol_p_pl # 1,1,K,D,D
        # para_Sigma_p_pl  # 1,1,K,L

        for rotate_angle in rotate_angles[:-1]:
            if False:  # random orthogonal
                A = rand_mat(dim=x_dim, method='orthogonal').to(device)  # rotation orthogonal  D,D
            else:  # sequential rotation
                rotate_theta = np.radians(rotate_angle)
                A = torch.tensor([
                    [np.cos(rotate_theta), np.sin(rotate_theta)],
                    [-np.sin(rotate_theta), np.cos(rotate_theta)],
                ], dtype=torch.float32).to(device)

            bar_mu_p_pl = torch.matmul(mu_p_pl, A.t())  # M1,M2,K,D
            bar_Sigma_p_pl = torch.matmul(A.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                                          torch.matmul(Sigma_p_pl, A.t()))  # 1,1,K,D,D
            # bar_Lchol_p_pl = torch.matmul(A.unsqueeze(0).unsqueeze(0).unsqueeze(0), Lchol_p_pl)  # 1,1,K,D,D
            bar_mu_q_pl = torch.matmul(mu_q_pl, A.t())  # M1,M2,K,D
            bar_Sigma_q_pl = torch.matmul(A.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                                          torch.matmul(Sigma_q_pl, A.t()))  # 1,1,K,D,D

            bar_samples_p_pl = torch.matmul(samples_p, A.t())  # M1,M2,N,D

            log_gmm_p, _, _ = GMM_logprobs_Full_pl(bar_samples_p_pl[:, :, :, :1], pi_p_pl, bar_mu_p_pl[:, :, :, :1],
                                                   bar_Sigma_p_pl[:, :, :, :1, :1])  # M1,M2,N
            log_gmm_q, _, _ = GMM_logprobs_Full_pl(bar_samples_p_pl[:, :, :, :1], pi_q_pl, bar_mu_q_pl[:, :, :, :1],
                                                   bar_Sigma_q_pl[:, :, :, :1, :1])  # M1,M2,N

            bar_KL_test_m1 = (log_gmm_p - log_gmm_q).mean(-1)
            bar_KL_test = bar_KL_test + bar_KL_test_m1

            ax = plt.figure().add_subplot(projection='3d')
            # ax.plot_surface(grid_mu1.cpu(), grid_mu2.cpu(), bar_KL_test_m1.cpu(), cmap=plt.cm.YlGnBu_r)
            ax.plot_surface(grid_mu1.cpu(), grid_mu2.cpu(), bar_KL_test_m1.cpu(), cmap='coolwarm')
            ax.set(xlim=(-2, 2), ylim=(-2, 2),  # zlim=(0, KL_test_m1.max()),
                   xlabel='$\mu_1$', ylabel='$\mu_2$', zlabel='Trans-Margin Reverse KL')
            # ax.scatter(1, -1, 0, color='r', marker="*")
            ax.xaxis.label.set_size(20)
            ax.yaxis.label.set_size(20)
            ax.zaxis.label.set_size(16)
            # plt.plot(-1, 1, 0, 'r', marker="*", markersize=20, zorder=5)
            # plt.plot(1, -1, 0, 'r', marker="*", markersize=20, zorder=5)
            if do_savefig:
                plt.savefig(out_dir + 'TranMarginRKL_RA%d'%(rotate_angle) + '_mesh.pdf', dpi=300)
                plt.savefig(out_dir + 'TranMarginRKL_RA%d'%(rotate_angle) + '_mesh.png', dpi=300)
            if do_pltshow:
                plt.show()
            else:
                plt.close()

            # eng.mesh(np.ascontiguousarray(grid_mu1),
            #          np.ascontiguousarray(grid_mu2),
            #          np.ascontiguousarray(bar_KL_test_m1))

            plt.pcolormesh(grid_mu1.cpu(), grid_mu2.cpu(), bar_KL_test_m1.abs().cpu(),
                           norm=colors.PowerNorm(gamma=gamma_color), cmap='coolwarm')
            plt.plot(-1, 1, 'r', marker="*", markersize=20, zorder=5)
            plt.plot(1, -1, 'r', marker="*", markersize=20, zorder=5)
            plt.xlabel('$\mu_1$', fontsize=20), plt.ylabel('$\mu_2$', fontsize=20)
            plt.axis('square')
            cbar = plt.colorbar()
            cbar.mappable.set_clim(0, max_colorbar)
            cbar.remove()
            if do_savefig:
                plt.savefig(out_dir + 'TranMarginRKL_RA%d'%(rotate_angle) + '_meshxy.pdf', dpi=300, bbox_inches='tight')
                plt.savefig(out_dir + 'TranMarginRKL_RA%d'%(rotate_angle) + '_meshxy.png', dpi=300, bbox_inches='tight')
            if do_pltshow:
                plt.show()
            else:
                plt.close()

            contourplot_gmm(pi_q, mu_q, Sigma_q)
            # plt.plot(np.array([0, A[0, 0]]), np.array([0, A[0, 1]]), 'r->')
            # plt.plot(np.array([0, A[1, 0]]), np.array([0, A[1, 1]]), 'b->')
            plt.arrow(0, 0, A[0, 0], A[0, 1], color='r', head_width=0.1, width=0.02)
            plt.xlabel('$x_1$', fontsize=20), plt.ylabel('$x_2$', fontsize=20)
            plt.axis('square')
            plt.axis([-2,2,-2,2])
            if rotate_angle == 0:
                pass
            elif rotate_angle < 90:
                plt.text(A[0, 0] + 0.3 * np.sign(A[0, 0]), A[0, 1] + 0.3 * np.sign(A[0, 1]), "$y_1$", fontsize=20)
            elif rotate_angle == 90:
                pass
            else:
                plt.text(A[0, 0] + 0.3 * np.sign(A[0, 0]), A[0, 1] + 0.3 * np.sign(A[0, 1]), "$y_2$", fontsize=20)
            # plt.plot(-1, 1, 'r', marker="*", markersize=10)
            plt.gcf().subplots_adjust(bottom=0.15)
            if do_savefig:
                plt.savefig(out_dir + 'TranMarginRKL_RA%d'%(rotate_angle) + '_proj.pdf', dpi=300, bbox_inches='tight')
                plt.savefig(out_dir + 'TranMarginRKL_RA%d'%(rotate_angle) + '_proj.png', dpi=300, bbox_inches='tight')
            if do_pltshow:
                plt.show()
            else:
                plt.close()

            inds = torch.linspace(-2, 2, 100).unsqueeze(-1)  # N,1
            QXY, _, _ = GMM_logprobs_Full(inds, pi_q, bar_mu_q_pl[0, 0, :, :1],
                                          bar_Sigma_q_pl[0, 0, :, :1, :1])  # K,1   K,D   K,D,D  N,1
            QXY = QXY.exp()

            plt.plot(inds, QXY)
            # plt.xlabel('$y_1$', fontsize=20), plt.ylabel('Trans-Margin PDF', fontsize=15)
            if rotate_angle == 0:
                plt.xlabel('$x_1$', fontsize=20), plt.ylabel('p($x_1$)', fontsize=20)
            elif rotate_angle < 90:
                plt.xlabel('$y_1$', fontsize=20), plt.ylabel('p($y_1$)', fontsize=20)
            elif rotate_angle == 90:
                plt.xlabel('$x_2$', fontsize=20), plt.ylabel('p($x_2$)', fontsize=20)
            else:
                plt.xlabel('$y_2$', fontsize=20), plt.ylabel('p($y_2$)', fontsize=20)
            plt.gcf().subplots_adjust(bottom=0.15)
            if do_savefig:
                plt.savefig(out_dir + 'TranMarginRKL_RA%d'%(rotate_angle) + '_margin_q1.pdf', dpi=300, bbox_inches='tight')
                plt.savefig(out_dir + 'TranMarginRKL_RA%d'%(rotate_angle) + '_margin_q1.png', dpi=300, bbox_inches='tight')
            if do_pltshow:
                plt.show()
            else:
                plt.close()

            # plt.scatter(x_all[:,0], x_all[:, 1], s=5, c=y_all)
            # plt.xlim(-2.5, 2.5),plt.ylim(-1.5, 1.5)
            # plt.plot(np.array([0, A[0,0]]), np.array([0, A[0,1]]), 'r-')
            # plt.plot(np.array([0, A[1,0]]), np.array([0, A[1,1]]), 'b-')
            # plt.title('training data')
            # plt.show()

    ax = plt.figure().add_subplot(projection='3d')
    # ax.plot_surface(grid_mu1.cpu(), grid_mu2.cpu(), bar_KL_test.cpu(), cmap=plt.cm.YlGnBu_r)
    ax.plot_surface(grid_mu1.cpu(), grid_mu2.cpu(), bar_KL_test.cpu(), cmap='coolwarm')
    ax.set(xlim=(-2, 2), ylim=(-2, 2),  # zlim=(0, KL_test_m1.max()),
           xlabel='$\mu_1$', ylabel='$\mu_2$', zlabel='Sum Trans-Margin Reverse KL')
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.zaxis.label.set_size(15)
    if do_savefig:
        plt.savefig(out_dir + 'SumTranMarginRKL_mesh.pdf', dpi=300)
        plt.savefig(out_dir + 'SumTranMarginRKL_mesh.png', dpi=300)
    if do_pltshow:
        plt.show()
    else:
        plt.close()

    plt.pcolormesh(grid_mu1.cpu(), grid_mu2.cpu(), bar_KL_test.abs().cpu(),
                   norm=colors.PowerNorm(gamma=gamma_color), cmap='coolwarm')
    plt.plot(-1, 1, 'r', marker="*", markersize=20, zorder=5)
    plt.plot(1, -1, 'r', marker="*", markersize=20, zorder=5)
    plt.xlabel('$\mu_1$', fontsize=20), plt.ylabel('$\mu_2$', fontsize=20)
    plt.axis('square')
    cbar = plt.colorbar()
    # cbar.mappable.set_clim(0, max_colorbar)
    cbar.remove()
    if do_savefig:
        plt.savefig(out_dir + 'SumTranMarginRKL_meshxy.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(out_dir + 'SumTranMarginRKL_meshxy.png', dpi=300, bbox_inches='tight')
    if do_pltshow:
        plt.show()
    else:
        plt.close()

## --------------------   Conditional Reverse KL loss    -------------------------------
if False:
    print('Processing Conditional Reverse KL ...')

    numX0 = numX1 = 7
    seed_torch(random_seed)

    [x0max, x1max], ind = x_test.max(0)
    [x0min, x1min], ind = x_test.min(0)
    x0min, x0max = x0min.round() + 0.1, x0max.round() - 0.1
    x1min, x1max = x1min.round() + 0.1, x1max.round() - 0.1
    x0s = torch.linspace(x0min, x0max, numX0)
    x1s = torch.linspace(x1min, x1max, numX0)

    #  x1 | x0
    with torch.no_grad():
        for i in range(numX0):

            # condition q params
            mu_q_pl_0 = mu_q_pl[:, :, :, :1]  # 1,1,K,D
            Sigma_q_pl_0 = Sigma_q_pl[:, :, :, :1, :1]  # 1,1,K,D,D

            x0 = x0s[i].expand(1, 1, 1, 1)

            log_gmm_q0, log_piNormal_q0, _ = GMM_logprobs_Full_pl(x0, pi_q_pl, mu_q_pl_0, Sigma_q_pl_0)  # 1,1,N,K
            pi_q_pl_10 = (log_piNormal_q0 - log_gmm_q0.unsqueeze(-1)).exp().permute(0, 1, 3, 2)  # 1,1,1,K

            mu_q_pl_10 = mu_q_pl[:, :, :, 1:]  # 1,1,K,D
            Sigma_q_pl_10 = Sigma_q_pl[:, :, :, 1:, 1:]  # 1,1,K,D,D

            # condition p params
            mu_p_pl_0 = mu_p_pl[:, :, :, :1]  # M1,M2,K,D
            Sigma_p_pl_0 = Sigma_p_pl[:, :, :, :1, :1]  # 1,1,K,D,D

            log_gmm_p0, log_piNormal_p0, _ = GMM_logprobs_Full_pl(x0, pi_p_pl, mu_p_pl_0, Sigma_p_pl_0)  # M1,M2,N,K
            pi_p_pl_10 = (log_piNormal_p0 - log_gmm_p0.unsqueeze(-1)).exp().permute(0, 1, 3, 2)  # M1,M2,1,K

            mu_p_pl_10 = mu_p_pl[:, :, :, 1:]  # M1,M2,K,D
            Sigma_p_pl_10 = Sigma_p_pl[:, :, :, 1:, 1:]  # 1,1,K,D,D

            # p samples
            _, Lchol_p_pl_10 = Sigma2Lchol_pl(Sigma_p_pl_10)  # 1,1,K,L
            samples_p_10, _ = GMM_sample_n_paraLchol_pl(pi_p_pl_10, mu_p_pl_10, Lchol_p_pl_10, n)  # M1,M2,N,D

            # condition KL loss
            log_gmm_p_10, _, _ = GMM_logprobs_Full_pl(samples_p_10, pi_p_pl_10, mu_p_pl_10,
                                                      Sigma_p_pl_10)  # M1,M2,N
            log_gmm_q_10, _, _ = GMM_logprobs_Full_pl(samples_p_10, pi_q_pl_10, mu_q_pl_10,
                                                      Sigma_q_pl_10)  # M1,M2,N
            condKL_test_10 = (log_gmm_p_10 - log_gmm_q_10).mean(-1)

            ax = plt.figure().add_subplot(projection='3d')
            # ax.plot_surface(grid_mu1.cpu(), grid_mu2.cpu(), condKL_test_10.cpu(), cmap=plt.cm.YlGnBu_r)
            ax.plot_surface(grid_mu1.cpu(), grid_mu2.cpu(), condKL_test_10.cpu(), cmap='coolwarm')
            ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(condKL_test_10.min() - 0.1, condKL_test_10.max() + 0.5),
                   xlabel='$\mu_1$', ylabel='$\mu_2$', zlabel='Condition Reverse KL')
            ax.xaxis.label.set_size(20)
            ax.yaxis.label.set_size(20)
            ax.zaxis.label.set_size(16)
            if do_savefig:
                plt.savefig(out_dir + 'CondRKL10' + str(i) + '_mesh.pdf', dpi=300)
                plt.savefig(out_dir + 'CondRKL10' + str(i) + '_mesh.png', dpi=300)
            if do_pltshow:
                plt.show()
            else:
                plt.close()

            plt.pcolormesh(grid_mu1.cpu(), grid_mu2.cpu(), condKL_test_10.abs().cpu(),
                           norm=colors.PowerNorm(gamma=gamma_color), cmap='coolwarm')
            plt.plot(-1, 1, 'r', marker="*", markersize=20, zorder=5)
            plt.plot(1, -1, 'r', marker="*", markersize=20, zorder=5)
            plt.xlabel('$\mu_1$', fontsize=20), plt.ylabel('$\mu_2$', fontsize=20)
            cbar = plt.colorbar()
            cbar.mappable.set_clim(0, max_colorbar)
            cbar.remove()
            plt.axis('square'),
            if do_savefig:
                plt.savefig(out_dir + 'CondRKL10' + str(i) + '_meshxy.pdf', dpi=300, bbox_inches='tight')
                plt.savefig(out_dir + 'CondRKL10' + str(i) + '_meshxy.png', dpi=300, bbox_inches='tight')
            if do_pltshow:
                plt.show()
            else:
                plt.close()

            contourplot_gmm(pi_q, mu_q, Sigma_q)
            # plt.plot(np.array([x0s[i], x0s[i]]), np.array([x1min, x1max]), 'b-')
            plt.plot(np.array([x0s[i], x0s[i]]), np.array([-2, 2]), 'b-')
            plt.xlabel('$x_1$', fontsize=20), plt.ylabel('$x_2$', fontsize=20)
            plt.axis('square')
            plt.axis([-2,2,-2,2])
            plt.gcf().subplots_adjust(bottom=0.15)
            if do_savefig:
                plt.savefig(out_dir + 'CondRKL10' + str(i) + '_loc.pdf', dpi=300, bbox_inches='tight')
                plt.savefig(out_dir + 'CondRKL10' + str(i) + '_loc.png', dpi=300, bbox_inches='tight')
            if do_pltshow:
                plt.show()
            else:
                plt.close()

            inds = torch.linspace(-2, 2, 100).unsqueeze(-1)  # N,1
            QXY, _, _ = GMM_logprobs_Full(inds, pi_q_pl_10[0, 0, :, :], mu_q_pl_10[0, 0, :, :1],
                                          Sigma_q_pl_10[0, 0, :, :1, :1])  # K,1   K,D   K,D,D  N,1
            QXY = QXY.exp()
            plt.plot(inds, QXY)
            # plt.xlabel('$x_2$', fontsize=20), plt.ylabel('Condition PDF', fontsize=15)
            plt.xlabel('$x_2$', fontsize=20), plt.ylabel('p($x_2$|$x_1$)', fontsize=20)
            plt.gcf().subplots_adjust(bottom=0.15)
            if do_savefig:
                plt.savefig(out_dir + 'CondRKL10' + str(i) + '_pdf.pdf', dpi=300, bbox_inches='tight')
                plt.savefig(out_dir + 'CondRKL10' + str(i) + '_pdf.png', dpi=300, bbox_inches='tight')
            if do_pltshow:
                plt.show()
            else:
                plt.close()

    #  x0 | x1
    with torch.no_grad():
        for i in range(numX1):

            # condition q params
            mu_q_pl_1 = mu_q_pl[:, :, :, 1:]  # 1,1,K,D
            Sigma_q_pl_1 = Sigma_q_pl[:, :, :, 1:, 1:]  # 1,1,K,D,D

            x1 = x1s[i].expand(1, 1, 1, 1)

            log_gmm_q1, log_piNormal_q1, _ = GMM_logprobs_Full_pl(x1, pi_q_pl, mu_q_pl_1, Sigma_q_pl_1)  # 1,1,N,K
            pi_q_pl_01 = (log_piNormal_q1 - log_gmm_q1.unsqueeze(-1)).exp().permute(0, 1, 3, 2)  # 1,1,1,K

            mu_q_pl_01 = mu_q_pl[:, :, :, :1]  # 1,1,K,D
            Sigma_q_pl_01 = Sigma_q_pl[:, :, :, :1, :1]  # 1,1,K,D,D

            # condition p params
            mu_p_pl_1 = mu_p_pl[:, :, :, 1:]  # M1,M2,K,D
            Sigma_p_pl_1 = Sigma_p_pl[:, :, :, 1:, 1:]  # 1,1,K,D,D

            log_gmm_p1, log_piNormal_p1, _ = GMM_logprobs_Full_pl(x1, pi_p_pl, mu_p_pl_1, Sigma_p_pl_1)  # M1,M2,N,K
            pi_p_pl_01 = (log_piNormal_p1 - log_gmm_p1.unsqueeze(-1)).exp().permute(0, 1, 3, 2)  # M1,M2,1,K

            mu_p_pl_01 = mu_p_pl[:, :, :, :1]  # M1,M2,K,D
            Sigma_p_pl_01 = Sigma_p_pl[:, :, :, :1, :1]  # 1,1,K,D,D

            # p samples
            _, Lchol_p_pl_01 = Sigma2Lchol_pl(Sigma_p_pl_01)  # 1,1,K,L
            samples_p_01, _ = GMM_sample_n_paraLchol_pl(pi_p_pl_01, mu_p_pl_01, Lchol_p_pl_01, n)  # M1,M2,N,D

            # condition KL loss
            log_gmm_p_01, _, _ = GMM_logprobs_Full_pl(samples_p_01, pi_p_pl_01, mu_p_pl_01,
                                                      Sigma_p_pl_01)  # M1,M2,N
            log_gmm_q_01, _, _ = GMM_logprobs_Full_pl(samples_p_01, pi_q_pl_01, mu_q_pl_01,
                                                      Sigma_q_pl_01)  # M1,M2,N
            condKL_test_01 = (log_gmm_p_01 - log_gmm_q_01).mean(-1)

            ax = plt.figure().add_subplot(projection='3d')
            # ax.plot_surface(grid_mu1.cpu(), grid_mu2.cpu(), condKL_test_01.cpu(), cmap=plt.cm.YlGnBu_r)
            ax.plot_surface(grid_mu1.cpu(), grid_mu2.cpu(), condKL_test_01.cpu(), cmap='coolwarm')
            ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(condKL_test_01.min() - 0.1, condKL_test_01.max() + 0.5),
                   xlabel='$\mu_1$', ylabel='$\mu_2$', zlabel='Condition Reverse KL')
            ax.xaxis.label.set_size(20)
            ax.yaxis.label.set_size(20)
            ax.zaxis.label.set_size(16)
            if do_savefig:
                plt.savefig(out_dir + 'CondRKL01' + str(i) + '_mesh.pdf', dpi=300)
                plt.savefig(out_dir + 'CondRKL01' + str(i) + '_mesh.png', dpi=300)
            if do_pltshow:
                plt.show()
            else:
                plt.close()

            plt.pcolormesh(grid_mu1.cpu(), grid_mu2.cpu(), condKL_test_01.abs().cpu(),
                           norm=colors.PowerNorm(gamma=gamma_color), cmap='coolwarm')
            plt.plot(-1, 1, 'r', marker="*", markersize=20, zorder=5)
            plt.plot(1, -1, 'r', marker="*", markersize=20, zorder=5)
            plt.xlabel('$\mu_1$', fontsize=20), plt.ylabel('$\mu_2$', fontsize=20)
            cbar = plt.colorbar()
            cbar.mappable.set_clim(0, max_colorbar)
            cbar.remove()
            plt.axis('square'),
            if do_savefig:
                plt.savefig(out_dir + 'CondRKL01' + str(i) + '_meshxy.pdf', dpi=300, bbox_inches='tight')
                plt.savefig(out_dir + 'CondRKL01' + str(i) + '_meshxy.png', dpi=300, bbox_inches='tight')
            if do_pltshow:
                plt.show()
            else:
                plt.close()

            contourplot_gmm(pi_q, mu_q, Sigma_q)
            # plt.plot(np.array([x0min, x0max]), np.array([x1s[i], x1s[i]]), 'b-')
            plt.plot(np.array([-2, 2]), np.array([x1s[i], x1s[i]]), 'b-')
            plt.xlabel('$x_1$', fontsize=20), plt.ylabel('$x_2$', fontsize=20)
            plt.axis('square')
            plt.axis([-2,2,-2,2])
            plt.gcf().subplots_adjust(bottom=0.15)
            if do_savefig:
                plt.savefig(out_dir + 'CondRKL01' + str(i) + '_loc.pdf', dpi=300, bbox_inches='tight')
                plt.savefig(out_dir + 'CondRKL01' + str(i) + '_loc.png', dpi=300, bbox_inches='tight')
            if do_pltshow:
                plt.show()
            else:
                plt.close()

            inds = torch.linspace(-2, 2, 100).unsqueeze(-1)  # N,1
            QXY, _, _ = GMM_logprobs_Full(inds, pi_q_pl_01[0, 0, :, :], mu_q_pl_01[0, 0, :, :1],
                                          Sigma_q_pl_01[0, 0, :, :1, :1])  # K,1   K,D   K,D,D  N,1
            QXY = QXY.exp()
            plt.plot(inds, QXY)
            # plt.xlabel('$x_1$', fontsize=20), plt.ylabel('Condition PDF', fontsize=15)
            plt.xlabel('$x_1$', fontsize=20), plt.ylabel('p($x_1$|$x_2$)', fontsize=20)
            plt.gcf().subplots_adjust(bottom=0.15)
            if do_savefig:
                plt.savefig(out_dir + 'CondRKL01' + str(i) + '_pdf.pdf', dpi=300, bbox_inches='tight')
                plt.savefig(out_dir + 'CondRKL01' + str(i) + '_pdf.png', dpi=300, bbox_inches='tight')
            if do_pltshow:
                plt.show()
            else:
                plt.close()

## --------------------   Tranformed Conditional Reverse KL loss    -------------------------------
if True:
    print('Processing Tranformed Conditional Reverse KL ...')

    numY1 = 7
    seed_torch(random_seed)

    rotate_angles = np.linspace(0, 90, 7)

    #  y0 | y1
    with torch.no_grad():
        # samples_p, m = GMM_sample_n_paraLchol_pl(pi_p_pl, mu_p_pl, Lchol_p_pl, n)  # M1,M2,N,D

        SumTcondKL_test = 0
        for rotate_angle in rotate_angles:

            rotate_theta = np.radians(rotate_angle)
            A = torch.tensor([
                [np.cos(rotate_theta), np.sin(rotate_theta)],
                [-np.sin(rotate_theta), np.cos(rotate_theta)],
            ], dtype=torch.float32).to(device)

            bar_mu_p_pl = torch.matmul(mu_p_pl, A.t())  # M1,M2,K,D
            bar_Sigma_p_pl = torch.matmul(A.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                                          torch.matmul(Sigma_p_pl, A.t()))  # 1,1,K,D,D

            bar_mu_q_pl = torch.matmul(mu_q_pl, A.t())  # 1,1,K,D
            bar_Sigma_q_pl = torch.matmul(A.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                                          torch.matmul(Sigma_q_pl, A.t()))  # 1,1,K,D,D

            # bar_samples_p_pl = torch.matmul(samples_p, A.t())  # M1,M2,N,D

            if False:
                bar_x_test = torch.matmul(x_test, A.t())  # M1,M2,K,D
                [y0max, y1max], ind = bar_x_test.max(0)
                [y0min, y1min], ind = bar_x_test.min(0)
                y1min, y1max = y1min.round() + 0.1, y1max.round() - 0.1
            else:
                tmp = torch.matmul(torch.tensor([
                    [1 + 3 * np.sqrt(sigma2), 3 * np.sqrt(sigma2)],
                    [1 + 3 * np.sqrt(sigma2), -3 * np.sqrt(sigma2)],
                    [-1 - 3 * np.sqrt(sigma2), 3 * np.sqrt(sigma2)],
                    [-1 - 3 * np.sqrt(sigma2), -3 * np.sqrt(sigma2)],
                ], dtype=torch.float32).to(device), A.t())  # 1,D
                y1max = tmp[:, 1].max()
                y1min = -y1max
            y1s = torch.linspace(y1min, y1max, numY1)

            for i in range(numY1):

                # condition q params
                bar_mu_q_pl_1 = bar_mu_q_pl[:, :, :, 1:]  # 1,1,K,Ds
                bar_Sigma_q_pl_1 = bar_Sigma_q_pl[:, :, :, 1:, 1:]  # 1,1,K,Ds,Ds

                y1 = y1s[i].expand(1, 1, 1, 1)  # 1,1,N,Ds

                log_gmm_q1, log_piNormal_q1, _ = GMM_logprobs_Full_pl(y1, pi_q_pl, bar_mu_q_pl_1, bar_Sigma_q_pl_1)  # 1,1,N,K
                pi_q_pl_01 = (log_piNormal_q1 - log_gmm_q1.unsqueeze(-1)).exp().permute(0, 1, 3, 2)  # 1,1,N,K

                matTMP = torch.matmul(bar_Sigma_q_pl[:, :, :, :1, 1:],
                                      torch.inverse(bar_Sigma_q_pl_1))  # 1,1,K,Dt,Ds
                bar_mu_q_pl_01 = bar_mu_q_pl[:, :, :, :1] + torch.matmul(
                    matTMP,  # 1,1,K,Dt,Ds
                    (y1 - bar_mu_q_pl_1).unsqueeze(-1)  # (1,1,N,Ds) - (1,1,K,Ds)  # N=1,
                ).squeeze(-1)  # 1,1,K,Dt
                bar_Sigma_q_pl_01 = bar_Sigma_q_pl[:, :, :, :1, :1] - torch.matmul(
                    matTMP, # 1,1,K,Dt,Ds
                    bar_Sigma_q_pl[:, :, :, 1:, :1]  # 1,1,K,Ds,Dt
                )  # 1,1,K,Dt,Dt

                # condition p params
                bar_mu_p_pl_1 = bar_mu_p_pl[:, :, :, 1:]  # M1,M2,K,Ds
                bar_Sigma_p_pl_1 = bar_Sigma_p_pl[:, :, :, 1:, 1:]  # 1,1,K,Ds,Ds

                log_gmm_p1, log_piNormal_p1, _ = GMM_logprobs_Full_pl(y1, pi_p_pl, bar_mu_p_pl_1, bar_Sigma_p_pl_1)  # M1,M2,N,K
                pi_p_pl_01 = (log_piNormal_p1 - log_gmm_p1.unsqueeze(-1)).exp().permute(0, 1, 3, 2)  # M1,M2,N,K

                matTMP = torch.matmul(bar_Sigma_p_pl[:, :, :, :1, 1:],
                                      torch.inverse(bar_Sigma_p_pl_1))  # 1,1,K,Dt,Ds
                bar_mu_p_pl_01 = bar_mu_p_pl[:, :, :, :1] + torch.matmul(
                    matTMP,  # 1,1,K,Dt,Ds
                    (y1 - bar_mu_p_pl_1).unsqueeze(-1)   # (1,1,N,Ds) - (M1,M2,K,Ds)  # N=1,
                ).squeeze(-1)  # M1,M2,K,Dt
                bar_Sigma_p_pl_01 = bar_Sigma_p_pl[:, :, :, :1, :1] - torch.matmul(
                    matTMP, # 1,1,K,Dt,Ds
                    bar_Sigma_p_pl[:, :, :, 1:, :1]  # 1,1,K,Ds,Dt
                )  # 1,1,K,Dt,Dt

                # p samples
                _, bar_Lchol_p_pl_01 = Sigma2Lchol_pl(bar_Sigma_p_pl_01)  # 1,1,K,L
                samples_p_01, _ = GMM_sample_n_paraLchol_pl(pi_p_pl_01, bar_mu_p_pl_01, bar_Lchol_p_pl_01, n)  # M1,M2,N,D

                # condition KL loss
                log_gmm_p_01, _, _ = GMM_logprobs_Full_pl(samples_p_01, pi_p_pl_01, bar_mu_p_pl_01,
                                                          bar_Sigma_p_pl_01)  # M1,M2,N
                log_gmm_q_01, _, _ = GMM_logprobs_Full_pl(samples_p_01, pi_q_pl_01, bar_mu_q_pl_01,
                                                          bar_Sigma_q_pl_01)  # M1,M2,N
                TcondKL_test_01 = (log_gmm_p_01 - log_gmm_q_01).mean(-1)
                SumTcondKL_test = SumTcondKL_test + TcondKL_test_01

                ax = plt.figure().add_subplot(projection='3d')
                # ax.plot_surface(grid_mu1.cpu(), grid_mu2.cpu(), TcondKL_test_01.cpu(), cmap=plt.cm.YlGnBu_r)
                ax.plot_surface(grid_mu1.cpu(), grid_mu2.cpu(), TcondKL_test_01.cpu(), cmap='coolwarm')
                ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(TcondKL_test_01.min() - 0.1, TcondKL_test_01.max() + 0.5),
                       xlabel='$\mu_1$', ylabel='$\mu_2$', zlabel='Trans-Condition Reverse KL')
                ax.xaxis.label.set_size(20)
                ax.yaxis.label.set_size(20)
                ax.zaxis.label.set_size(16)
                if do_savefig:
                    plt.savefig(out_dir + 'TransCondRKL_RA%d'%(rotate_angle) + '_01' + str(i) + '_mesh.pdf', dpi=300)
                    plt.savefig(out_dir + 'TransCondRKL_RA%d'%(rotate_angle) + '_01' + str(i) + '_mesh.png', dpi=300)
                if do_pltshow:
                    plt.show()
                else:
                    plt.close()

                plt.pcolormesh(grid_mu1.cpu(), grid_mu2.cpu(), TcondKL_test_01.abs().cpu(),
                               norm=colors.PowerNorm(gamma=gamma_color), cmap='coolwarm')
                plt.plot(-1, 1, 'r', marker="*", markersize=20, zorder=5)
                plt.plot(1, -1, 'r', marker="*", markersize=20, zorder=5)
                plt.xlabel('$\mu_1$', fontsize=20), plt.ylabel('$\mu_2$', fontsize=20)
                cbar = plt.colorbar()
                cbar.mappable.set_clim(0, max_colorbar)
                cbar.remove()
                plt.axis('square'),
                if do_savefig:
                    plt.savefig(out_dir + 'TransCondRKL_RA%d'%(rotate_angle) + '_01' + str(i) + '_meshxy.pdf', dpi=300, bbox_inches='tight')
                    plt.savefig(out_dir + 'TransCondRKL_RA%d'%(rotate_angle) + '_01' + str(i) + '_meshxy.png', dpi=300, bbox_inches='tight')
                if do_pltshow:
                    plt.show()
                else:
                    plt.close()

                contourplot_gmm(pi_q, mu_q, Sigma_q)
                plt.arrow(0, 0, A[0, 0], A[0, 1], color='r', head_width=0.1, width=0.02)
                plt.text(A[0, 0] + 0.3 * np.sign(A[0, 0]), A[0, 1] + 0.3 * np.sign(A[0, 1]), "$y_1$", fontsize=20)
                plt.arrow(0, 0, A[1, 0], A[1, 1], color='r', head_width=0.1, width=0.02)
                plt.text(A[1, 0] + 0.3 * np.sign(A[1, 0]), A[1, 1] + 0.3 * np.sign(A[1, 1]), "$y_2$", fontsize=20)
                # x2ends = torch.tensor([
                #     [y0min, y1s[i]],
                #     [y0max, y1s[i]]
                # ]).mm(A)
                x2ends = torch.tensor([
                    [-2, y1s[i]],
                    [2, y1s[i]]
                ]).mm(A)
                plt.plot(x2ends[:, 0], x2ends[:, 1], 'b-')
                plt.xlabel('$x_1$', fontsize=20), plt.ylabel('$x_2$', fontsize=20)
                plt.axis('square')
                plt.axis([-2,2,-2,2])
                plt.gcf().subplots_adjust(bottom=0.15)
                if do_savefig:
                    plt.savefig(out_dir + 'TransCondRKL_RA%d'%(rotate_angle) + '_01' + str(i) + '_loc.pdf', dpi=300, bbox_inches='tight')
                    plt.savefig(out_dir + 'TransCondRKL_RA%d'%(rotate_angle) + '_01' + str(i) + '_loc.png', dpi=300, bbox_inches='tight')
                if do_pltshow:
                    plt.show()
                else:
                    plt.close()

                inds = torch.linspace(-2, 2, 100).unsqueeze(-1)  # N,1
                QXY, _, _ = GMM_logprobs_Full(inds, pi_q_pl_01[0, 0, :, :], bar_mu_q_pl_01[0, 0, :, :1],
                                              bar_Sigma_q_pl_01[0, 0, :, :1, :1])  # K,1   K,D   K,D,D  N,1
                QXY = QXY.exp()
                plt.plot(inds, QXY)
                # plt.xlabel('$y_1$', fontsize=20), plt.ylabel('Trans-Condition PDF', fontsize=15)
                plt.xlabel('$y_1$', fontsize=20), plt.ylabel('p($y_1$|$y_2$)', fontsize=20)
                plt.gcf().subplots_adjust(bottom=0.15)
                if do_savefig:
                    plt.savefig(out_dir + 'TransCondRKL_RA%d'%(rotate_angle) + '_01' + str(i) + '_pdf.pdf', dpi=300, bbox_inches='tight')
                    plt.savefig(out_dir + 'TransCondRKL_RA%d'%(rotate_angle) + '_01' + str(i) + '_pdf.png', dpi=300, bbox_inches='tight')
                if do_pltshow:
                    plt.show()
                else:
                    plt.close()

    ax = plt.figure().add_subplot(projection='3d')
    # ax.plot_surface(grid_mu1.cpu(), grid_mu2.cpu(), bar_KL_test.cpu(), cmap=plt.cm.YlGnBu_r)
    ax.plot_surface(grid_mu1.cpu(), grid_mu2.cpu(), SumTcondKL_test.cpu(), cmap='coolwarm')
    ax.set(xlim=(-2, 2), ylim=(-2, 2),  # zlim=(0, KL_test_m1.max()),
           xlabel='$\mu_1$', ylabel='$\mu_2$', zlabel='Sum Trans-Condition Reverse KL')
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.zaxis.label.set_size(15)
    if do_savefig:
        plt.savefig(out_dir + 'SumTranConditionRKL_mesh.pdf', dpi=300)
        plt.savefig(out_dir + 'SumTranConditionRKL_mesh.png', dpi=300)
    if do_pltshow:
        plt.show()
    else:
        plt.close()

    plt.pcolormesh(grid_mu1.cpu(), grid_mu2.cpu(), SumTcondKL_test.abs().cpu(),
                   norm=colors.PowerNorm(gamma=gamma_color), cmap='coolwarm')
    plt.plot(-1, 1, 'r', marker="*", markersize=20, zorder=5)
    plt.plot(1, -1, 'r', marker="*", markersize=20, zorder=5)
    plt.xlabel('$\mu_1$', fontsize=20), plt.ylabel('$\mu_2$', fontsize=20)
    plt.axis('square')
    cbar = plt.colorbar()
    # cbar.mappable.set_clim(0, max_colorbar)
    cbar.remove()
    if do_savefig:
        plt.savefig(out_dir + 'SumTranConditionRKL_meshxy.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(out_dir + 'SumTranConditionRKL_meshxy.png', dpi=300, bbox_inches='tight')
    if do_pltshow:
        plt.show()
    else:
        plt.close()

curr_time = time.strftime("%H:%M:%S", time.localtime())
print("Current Time is :", curr_time)

# eng.quit()
