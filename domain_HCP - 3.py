import os
import torch
import torch.nn as nn
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pretrain3

torch.set_default_dtype(torch.float64)  # 设置全局默认类型为float64
alpha1 = 10
alpha2 = 1
r1 = 1/7
OMEGA = 10


def exact_solution(x, y):
    Theta = torch.atan2(y, x)
    r = torch.sqrt(x**2 + y**2)
    inside = r <= (0.5 + r1 * torch.sin(OMEGA * Theta))
    res = torch.zeros_like(x)
    res[~inside] = (r[~inside]**4 + 0.1 * torch.log(2 * r[~inside])) / alpha1
    res[inside] = r[inside]**2 / alpha2
    return res


def f(x, y, area):
    r = torch.sqrt(x ** 2 + y ** 2)
    if area:
        res = -16 * r**2
    else:
        res = torch.tensor(-4.0) * torch.ones_like(x)
    return res

def g(x,y):
    r = torch.sqrt(x**2 + y**2)
    res = (r**4 + 0.1 * torch.log(2 * r)) / alpha1
    return res

def generate_polar_samples(N, r_min, r_max, device='cpu'):
    """在极坐标环形区域生成均匀采样点"""
    theta = torch.rand(N, 1) * 2 * torch.pi
    r = torch.sqrt(torch.rand(N, 1)) * (r_max - r_min) + r_min
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)

    return torch.cat([x, y], dim=1).to(device)



def generate_samples(N_omega, N_gamma, N_inside, N_outside, device='cpu'):
    """
    生成太阳花界面问题的采样点

    参数:
    - N_omega: 正方形边界采样点数 (必须能被4整除)
    - N_gamma: 太阳花界面采样点数
    - N_inside: 界面内部(太阳花内部)采样点数
    - N_outside: 界面外部(太阳花外部)采样点数
    - device: 计算设备

    返回:
    - 包含所有采样点的字典
    """
    samples = {}

    # 正方形边界采样 (X_omega)
    N_per_side = N_omega // 4
    # 左边界 (x=-1, y∈[-1,1])
    left = torch.cat([
        -torch.ones(N_per_side, 1),
        torch.rand(N_per_side, 1) * 2 - 1  # y ∈ [-1,1]
    ], dim=1)
    # 右边界 (x=1, y∈[-1,1])
    right = torch.cat([
        torch.ones(N_per_side, 1),
        torch.rand(N_per_side, 1) * 2 - 1
    ], dim=1)
    # 下边界 (y=-1, x∈[-1,1])
    bottom = torch.cat([
        torch.rand(N_per_side, 1) * 2 - 1,
        -torch.ones(N_per_side, 1)
    ], dim=1)
    # 上边界 (y=1, x∈[-1,1])
    top = torch.cat([
        torch.rand(N_per_side, 1) * 2 - 1,
        torch.ones(N_per_side, 1)
    ], dim=1)

    samples['X_omega'] = torch.cat([left, right, bottom, top], dim=0).to(device)

    # 太阳花界面采样 (X_gamma) --------------------------------------------
    theta = torch.rand(N_gamma, 1) * 2 * torch.pi
    r = 0.5 + r1 * torch.sin(OMEGA * theta) # 极坐标下的界面方程
    x_gamma = r * torch.cos(theta)
    y_gamma = r * torch.sin(theta)
    samples['X_gamma'] = torch.cat([x_gamma, y_gamma], dim=1).to(device)


    N1 = N_inside//2
    N2 = N_inside - N1
    x1 = torch.distributions.Beta(3, 1).sample((N2, 1))
    # 内部区域采样
    theta = torch.rand(N1, 1) * 2 * torch.pi
    r_max = 0.5 + r1 * torch.sin(OMEGA * theta)
    r = torch.sqrt(torch.rand(N1, 1)) * r_max

    r0 = x1 * r_max
    sample1 = torch.cat([r * torch.cos(theta), r * torch.sin(theta)], dim=1).to(device)
    sample2 = torch.cat([r0 * torch.cos(theta), r0 * torch.sin(theta)], dim=1).to(device)


    samples['X_inside2'] = torch.cat([sample1, sample2], dim=0)

    # 外部区域采样
    theta = torch.rand(N_outside, 1) * 2 * torch.pi
    r_min = 0.5 + r1 * torch.sin(OMEGA * theta)  # 太阳花边界
    r_max = torch.sqrt(torch.tensor(2.0))  # 最大半径

    # 生成
    r = torch.sqrt(torch.rand(N_outside, 1) * (r_max ** 2 - r_min ** 2) + r_min ** 2)

    # 转换为笛卡尔坐标
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)

    # 确保所有点都在 [-1, 1] × [-1, 1] 的正方形内
    x = torch.clamp(x, -1.0, 1.0)
    y = torch.clamp(y, -1.0, 1.0)

    samples['X_inside1'] = torch.cat([x, y], dim=1).to(device)
    return samples




# 定义神经网络结构
class PINN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=100, output_dim=1):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.net(x)


class FNN(nn.Module):

    def __init__(self, m=64, n=6, input_dim=2, output_dim=1, actv=nn.Tanh()):  # m每层神经元数，n为隐藏层数
        super(FNN, self).__init__()
        self.actv = actv
        # self.act_para0 = torch.nn.Parameter(torch.ones(1).requires_grad_())
        # self.act_para1 = torch.nn.Parameter(torch.ones(1).requires_grad_())
        self.linear_input = nn.Linear(input_dim, m)
        self.dense = torch.nn.Sequential()
        for i in range(n):
            self.dense.add_module('dense_layer', torch.nn.Linear(m, m))
            self.dense.add_module('dense_actv', self.actv)
        self.linear_output = nn.Linear(m, output_dim)

    def forward(self, x):
        y = self.actv(self.linear_input(x))
        y = self.dense(y)
        output = self.linear_output(y)
        return output


class observe_point():
    def __init__(self, N_omega=None, N_gamma=None, N_inside1=None, N_inside2=None):
        self.N_omega = N_omega
        self.N_gamma = N_gamma
        self.N_inside1 = N_inside1
        self.N_inside2 = N_inside2



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'using {device} to simulate')





rho = 0.5  # 相关系数
beta = 0.01  # 界面损失权重
max_outer_iter = 1 # 外层最大迭代次数
inner_epochs = 80000  # 内层最大迭代次数

N_omega = 100
N_gamma = 100
N_inside1 = 500
N_inside2 = 500


print('start  to select point')
samples = generate_samples(
    N_omega=N_omega,
    N_gamma=N_gamma,
    N_outside=N_inside1,
    N_inside=N_inside2,
    device=device
)

def plot_sunflower_samples(samples):
    """绘制太阳花界面问题的采样点"""
    plt.figure(figsize=(10, 10))

    # 绘制正方形边界点 (蓝色)
    if 'X_omega' in samples:
        plt.scatter(samples['X_omega'][:, 0].cpu(), samples['X_omega'][:, 1].cpu(),
                    c='blue', s=5, label='Boundary (Ω)')

    # 绘制太阳花界面点 (红色)
    if 'X_gamma' in samples:
        plt.scatter(samples['X_gamma'][:, 0].cpu(), samples['X_gamma'][:, 1].cpu(),
                    c='red', s=5, label='Interface (Γ)')

    # 绘制内部区域点 (绿色)
    if 'X_inside2' in samples:
        plt.scatter(samples['X_inside2'][:, 0].cpu(), samples['X_inside2'][:, 1].cpu(),
                    c='green', s=5, label='Inside (D2)')

    # 绘制外部区域点 (紫色)
    if 'X_inside1' in samples:
        plt.scatter(samples['X_inside1'][:, 0].cpu(), samples['X_inside1'][:, 1].cpu(),
                    c='purple', s=5, label='Outside (D1)')

    plt.title('Sunflower Interface Sampling Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # 保持纵横比相同
    plt.savefig('sunflower_samples.png')

plot_sunflower_samples(samples)

print('point has been selected')

observe_ = generate_samples(
    N_omega=100,
    N_gamma=100,
    N_outside=100,
    N_inside=100,
    device=device
)

train_point = observe_point()
train_point.N_omega = observe_['X_omega']
train_point.N_gamma = observe_['X_gamma']
train_point.N_inside1 = observe_['X_inside1']
train_point.N_inside2 = observe_['X_inside2']


# 打印各区域采样点数量
for region, points in samples.items():
    print(f"{region}: {points.shape[0]} points")

X_omega = samples['X_omega']                 # 外部边界点
X_gamma = samples['X_gamma']                 # 内部界面点
X_inside1 = samples['X_inside1']             # 圆外内点
X_inside2 = samples['X_inside2']             # 圆内内点
minloss = 1000000000000
minloss2 = 1e10
loss_old = 10000

# 初始化界面参数
phi = torch.zeros(N_gamma, 1)
Phi = torch.zeros(N_gamma, 1)



##################################################################################

seed = 50
deltaX = 0.01
deltaY = 0.01



# HCP 部分
def collocation_point_sim(results):
    col1 = results.clone()
    col2 = results.clone()
    col2[:, 0:1] = col1[:, 0:1] + deltaX/2
    col3 = results.clone()
    col3[:, 0:1] = col3[:, 0:1] - deltaX/2
    col4 = results.clone()
    col4[:, 1:2] = col4[:, 1:2] + deltaY/2
    col5 = results.clone()
    col5[:, 1:2] = col5[:, 1:2] - deltaY/2

    return torch.cat((col1, col2, col3, col4, col5), dim=1)


def buildA(x):
    # 构建A矩阵，AH=0，
    # 输出为A矩阵，用于后续映射
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = x.size()[0]
    Theta = torch.atan2(x[:, 1], x[:, 0])
    r = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)
    item = r <= (0.5 + r1 * torch.sin(OMEGA * Theta))
    item = item.to('cpu')
    A = np.zeros((batch_size, 5))
    alpha = np.ones_like(A) * alpha2
    alpha[~item] = alpha1
    A[:, 0:1] = -((alpha[:, 1:2] + alpha[:, 2:3])/deltaX ** 2 + (alpha[:, 3:4] + alpha[:, 4:5])/ deltaY ** 2)
    A[:, 1:2] = 1 / deltaX ** 2 * alpha[:, 1:2]
    A[:, 2:3] = 1 / deltaX ** 2 * alpha[:, 2:3]
    A[:, 3:4] = 1 / deltaY ** 2 * alpha[:, 3:4]
    A[:, 4:5] = 1 / deltaY ** 2 * alpha[:, 4:5]
    return torch.from_numpy(A).type(torch.DoubleTensor).to(device).unsqueeze(1)

def projection_batch(A, H, b):
    # 通用性映射函数，
    # 输入为A矩阵，H矩阵（模型预测结果）
    # 输出为H_矩阵，是映射后的模型预测结果
    norm = torch.max(A)
    A_norm = A / norm
    b_norm = b / norm
    I = torch.eye(A.size()[2]).double().to(device).repeat(A.size()[0], 1, 1)
    invs = torch.inverse(torch.bmm(A_norm, A_norm.permute(0, 2, 1)))
    M = torch.bmm(A_norm.permute(0, 2, 1), invs)
    intermediate = torch.bmm(M, A_norm)
    H_ = torch.bmm((I - intermediate), H) +  torch.bmm(M , b_norm)
    AH = torch.bmm(A, H)
    AH_ = torch.bmm(A, H_)
    hc_loss = torch.abs(AH-b)
    hc_loss_ = torch.abs(AH_-b)
    return H_.squeeze(2), hc_loss, hc_loss_

def HCP_observe_pre(indx, label, area):
    '''
    此函数用于输出观测值点的损失以及投影后预测的结果
    '''
    H = label.unsqueeze(2)
    A = buildA(indx)
    b = -f(indx[:,0:1], indx[:,1:2], area).unsqueeze(2)
    H_p, Hloss, H_ploss = projection_batch(A, H, b)
    h_pre_value = torch.mm(H_p, torch.tensor([[1], [0], [0], [0], [0]]).type(torch.DoubleTensor).to(device))  # 取第一列即原点的值
    return h_pre_value


def pred_data(u_net, X, area):
    col_point_i = collocation_point_sim(X).to(device)
    prei = torch.empty((col_point_i.size()[0], 5)).to(device)
    for _ in range(5):
        rs = u_net(col_point_i[:, _ * 2:_ * 2 + 2])
        prei[:, _] = rs.squeeze(1)
    # 计算PDE残差
    u = HCP_observe_pre(col_point_i, prei, area)  # 投影后的预测结果

    return u




def bound_adjust(indx, label):
    '''
    此函数用于二维填充边界配置点周围的点数值
    '''
    currentx1 = indx[:, 2:4]               # x+    1
    currentx2 = indx[:, 4:6]               # x-    2
    currenty1 = indx[:, 6:8]               # y+    3
    currenty2 = indx[:, 8:10]              # y-    4
    num = label.size(0)
    for _ in range(num):
        if currentx1[_, 0] > 1:            # x+ beyond 1
            label[_, 1] = (-f(indx[_, 0], indx[_, 1], 1) /alpha1 - (label[_, 3] + label[_, 4] - 2 * label[_, 0]) /deltaY **2) * deltaX **2 + 2 * label[_, 0] - label[_, 2]
        if currentx2[_, 0] < -1:            # x- lower -1
            label[_, 2] = (-f(indx[_, 0], indx[_, 1], 1) /alpha1 - (label[_, 3] + label[_, 4] - 2 * label[_, 0]) /deltaY **2) * deltaX **2 + 2 * label[_, 0] - label[_, 1]
        if currenty1[_, 1] > 1:
            label[_, 3] = (-f(indx[_, 0], indx[_, 1], 1) /alpha1 - (label[_, 1] + label[_, 2] - 2 * label[_, 0]) /deltaX **2) * deltaY **2 + 2 * label[_, 0] - label[_, 4]
        if currenty2[_, 1] < -1:
            label[_, 4] = (-f(indx[_, 0], indx[_, 1], 1) /alpha1 - (label[_, 1] + label[_, 2] - 2 * label[_, 0]) /deltaX **2) * deltaY **2 + 2 * label[_, 0] - label[_, 3]
    return label




def compute_interface_loss(u_net1,u_net2, X, phi, Phi, flag):
    '''

    :param u_net1:   外层的神经网络
    :param u_net2:   内层的神经网络
    :param X:        界面点
    :param phi:      界面条件
    :param Phi:
    :param flag:     flag= 1 条件1 否则条件0
    :return:         界面损失
    '''
    X.requires_grad_(True)
    u1 = pred_data(u_net1, X, 1)
    u2 = pred_data(u_net2, X, 0)
    grad_u1 = torch.autograd.grad(u1[:, 0].sum(), X, create_graph=True)[0]
    grad_u2 = torch.autograd.grad(u2[:, 0].sum(), X, create_graph=True)[0]
    if flag:
        theta = torch.atan2(X[:, 0], X[:, 1])
        r = 0.5 + r1 * torch.sin(OMEGA * theta)
        dr_dtheta = r1 * OMEGA * torch.cos(OMEGA * theta)
        # 计算切向量分量 (dx/dθ, dy/dθ)
        dx_dtheta = dr_dtheta * torch.cos(theta) - r * torch.sin(theta)
        dy_dtheta = dr_dtheta * torch.sin(theta) + r * torch.cos(theta)

        # 构造法向量 (n_x, n_y) = (dy/dθ, -dx/dθ)
        n_x = dy_dtheta
        n_y = -dx_dtheta

        # 归一化
        norm = torch.sqrt(n_x ** 2 + n_y ** 2)
        n_x = n_x / norm
        n_y = n_y / norm
        loss = alpha1 * (grad_u1[:, 0:1] * n_x + grad_u1[:, 1:2] * n_y) - alpha2 *( grad_u2[:, 0:1] * n_x + grad_u2[:, 1:2] * n_y)  - Phi
    else:
        loss = u1-u2-phi
    return torch.mean((loss) ** 2)


def compute_boundary_loss(u_net, X, area):
    u = pred_data(u_net,X, area)
    return torch.mean((u - g(X[:, 0:1], X[:, 1:2]))**2)

def obs_loss(u_net, X, area):
    u = pred_data(u_net,X, area)
    realdata = exact_solution(X[:, 0:1], X[:, 1:2])
    return torch.mean((u - realdata) ** 2)


def compute_pde_loss(net, X, area):
    if area:
        alpha = alpha1
    else:
        alpha = alpha2
    u = pred_data(net, X, area)

    grad_u = torch.autograd.grad(u.sum(), X, create_graph=True)[0]
    grad_u_x, grad_u_y = grad_u[:, 0:1], grad_u[:, 1:2]

    grad_u_x2 = torch.autograd.grad(grad_u_x.sum(), X, create_graph=True)[0][:, 0:1]
    grad_u_y2 = torch.autograd.grad(grad_u_y.sum(), X, create_graph=True)[0][:, 1:2]
    delta_u = grad_u_x2 + grad_u_y2

    F = f(X[:, 0:1], X[:, 1:2], area)
    pde_loss = torch.mean((alpha * delta_u + F) ** 2)

    return pde_loss

# 定义损失函数
def compute_loss(u1_net, u2_net, X_omega, X_d, X_gamma, phi, Phi , area):
    # 启用梯度计算
    X_omega.requires_grad_(True)
    X_d = X_d.to(device)
    X_omega  = X_omega.to(device)
    X_gamma = X_gamma.to(device)
    Phi = Phi.to(device)
    phi = phi.to(device)

    if area:
        pde_loss = compute_pde_loss(u1_net, X_omega, area)
        # 边界损失
        d_loss = compute_boundary_loss(u1_net, X_d, area)

        # # 内部观测损失
        obs_gamma_loss = obs_loss(u1_net, train_point.N_inside1, area)

        interface_loss = compute_interface_loss(u1_net,u2_net, X_gamma, phi, Phi, 1)

        total_loss = pde_loss + 20 * obs_gamma_loss + 40 * d_loss + beta * interface_loss


    else:
        pde_loss = compute_pde_loss(u2_net, X_omega, area)
        # 观测损失
        obs_gamma_loss = obs_loss(u2_net, train_point.N_inside2, area)

        interface_loss = compute_interface_loss(u1_net, u2_net, X_gamma, phi, Phi, 0)


        total_loss = pde_loss + 20 * obs_gamma_loss + beta * interface_loss



    return total_loss, pde_loss, interface_loss
########################################################################################
# 主循环
u1_net = PINN()
u2_net = PINN()
u1_net.to(device)
u2_net.to(device)
optimizer1 = torch.optim.Adam(u1_net.parameters(), lr=1e-3)
optimizer2 = torch.optim.Adam(u2_net.parameters(), lr=1e-3)

# 变量存放到GPU
phi = phi.to(device)
Phi = Phi.to(device)

minloss = 1000000000000
# loss_old = 100000000
#
u1_net.load_state_dict(pretrain3.best_model1)
u2_net.load_state_dict(pretrain3.best_model2)

B = torch.eye(3)
B[0][0] = 0

for k in range(max_outer_iter):
    print(f"\nOuter Iteration {k + 1}/{max_outer_iter}")

    # 训练子问题2
    for epoch in range(inner_epochs):
        optimizer2.zero_grad()
        total_loss2, pdeloss2, interfaceloss2 = compute_loss(u1_net, u2_net, X_inside2, X_omega, X_gamma, phi, Phi, 0)
        total_loss2.backward()
        optimizer2.step()
        if epoch % 100 == 0:
            print(f"Subproblem 2 - Epoch {epoch}: Loss = {total_loss2.item():.4e}, pdeLoss = {pdeloss2.item():.4e}, interfaceLoss = {interfaceloss2.item():.4e}")

    realdata2 = exact_solution(X_inside2[:, 0].detach().to('cpu'), X_inside2[:, 1].detach().to('cpu'))
    u2 = pred_data(u2_net, X_inside2, 0)
    real_loss2 = torch.norm(u2[:, 0] - realdata2.clone().detach().to(device), 2)/ (torch.norm(realdata2, 2))

    if total_loss2 < minloss2:
        minloss2 = total_loss2
        best_model2 = copy.deepcopy(u2_net.state_dict())

    with torch.no_grad():  # 全局禁用梯度
        # 局部启用梯度计算
        with torch.enable_grad():
            u2_gamma = pred_data(u2_net, X_gamma, 0)
            u2_grad = torch.autograd.grad(u2_gamma[:, 0].sum(), X_gamma, create_graph=True)[0]

        theta = torch.atan2(X_gamma[:, 0], X_gamma[:, 1])
        r = 0.5 + r1 * torch.sin(OMEGA * theta)
        dr_dtheta = r1 * OMEGA * torch.cos(OMEGA * theta)
        # 计算切向量分量
        dx_dtheta = dr_dtheta * torch.cos(theta) - r * torch.sin(theta)
        dy_dtheta = dr_dtheta * torch.sin(theta) + r * torch.cos(theta)

        # 构造法向量
        n_x = dy_dtheta
        n_y = -dx_dtheta

        # 归一化
        norm = torch.sqrt(n_x ** 2 + n_y ** 2)
        n_x = n_x / norm
        n_y = n_y / norm

        Phi = - alpha2 * (u2_grad[:, 0:1] * n_x + u2_grad[:, 1:2] * n_y)


    # 训练子问题1
    for epoch in range(inner_epochs):
        optimizer1.zero_grad()
        total_loss1, pdeloss1, interfaceloss1 = compute_loss(u1_net, u2_net, X_inside1,X_omega, X_gamma, phi, Phi, 1)
        _, l2, _ = compute_loss(u1_net, u2_net, X_inside2, X_omega, X_gamma, phi, Phi, 1)
        total_loss1 += l2
        total_loss1.backward()
        optimizer1.step()
        if epoch % 100 == 0:
            print(f"Subproblem 1 - Epoch {epoch}: Loss = {total_loss1.item():.4e},pdeLoss = {pdeloss1.item():.4e}, interfaceLoss = {interfaceloss1.item():.4e}")

    realdata1 = exact_solution(X_inside1[:, 0].detach().to('cpu'), X_inside1[:, 1].detach().to('cpu'))
    batch_size = X_inside1.shape[0]
    B_batch = B.unsqueeze(0).expand(batch_size, -1, -1).to(device)
    u1 = pred_data(u1_net, X_inside1, 1)
    real_loss1 = torch.norm(u1[:, 0] - realdata1.clone().detach().to(device), 2)/ (torch.norm( realdata1, 2))

    if total_loss1 < minloss:
        minloss = total_loss1
        best_model1 = copy.deepcopy(u1_net.state_dict())


    # 更新接口条件
    with torch.no_grad():
        u1_gamma = pred_data(u1_net, X_gamma, 1)
        u2_gamma = pred_data(u2_net, X_gamma, 0)
        phi = rho * u2_gamma + (1 -rho) * u1_gamma


    u1_net.load_state_dict(best_model1)
    u2_net.load_state_dict(best_model2)


    u1 = pred_data(u1_net, X_inside1, 1)
    u2 = pred_data(u2_net, X_inside2, 0)

    real_loss1 = torch.norm((u1[:, 0] - realdata1.clone().detach().to(device)), 2) / (torch.norm((realdata1), 2))

    real_loss2 = torch.norm((u2[:, 0] - realdata2.clone().detach().to(device)), 2) / (torch.norm((realdata2), 2))

    print(f'realloss1 = {real_loss1}, realloss2 = {real_loss2}')


    ################################################################
    # 采样点用于测试

    # samples_test = generate_samples(
    #     N_omega=500,
    #     N_gamma=500,
    #     N_inside1=1250,
    #     N_inside2=1250,
    #     device=device
    # )
    #
    # ind1 = samples_test['X_inside1']
    # ind2 = samples_test['X_inside2']
    # u1_ = pred_data(u1_net, ind1)
    # u2_ = pred_data(u2_net, ind2)
    # realdata1_ = exact_solution(ind1[:, 0].detach().to('cpu'), ind1[:, 1].detach().to('cpu'))
    # realdata2_ = exact_solution(ind2[:, 0].detach().to('cpu'), ind2[:, 1].detach().to('cpu'))
    #
    # real_samp1 = torch.norm(u1_[:, 0] - realdata1_.clone().detach().to(device) , 2)/ (torch.norm((realdata1_), 2))
    # real_samp2 = torch.norm(u2_[:, 0] - realdata2_.clone().detach().to(device), 2) / (torch.norm((realdata2_), 2))
    #
    # real_loss = torch.norm((u1_[:, 0] - realdata1_.clone().detach().to(device) +
    #                         u2_[:, 0] - realdata2_.clone().detach().to(device))
    # , 2)/ (torch.norm((realdata2_ + realdata1_), 2))
    # print(f'train_samples_realloss1 = {real_samp1},train_samples_realloss2 = {real_samp2},entire_loss = {real_loss}')
    #

def plot_results(u1_net, u2_net):
    # 将模型移到CPU并设置为评估模式
    n_points = 200  # 网格分辨率

    # 生成测试网格 [-1,1],[-1,1]
    xx, yy = torch.meshgrid(torch.linspace(-1, 1, n_points),
                            torch.linspace(-1, 1, n_points))
    X_test = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)


    # 计算太阳花界面掩码
    r = torch.sqrt(X_test[:, 0] ** 2 + X_test[:, 1] ** 2)
    theta = torch.atan2(X_test[:, 1], X_test[:, 0])
    mask = r <= (0.5 + r1 * torch.sin(OMEGA * theta))

    # 计算预测解
    with torch.no_grad():
        pred = torch.zeros_like(r)
        pred[mask] = pred_data(u2_net, X_test[mask].to(device), 0).squeeze().cpu()  # 内部区域
        exact = exact_solution(X_test[:, 0], X_test[:, 1])
        pred[~mask] = pred_data(u1_net, X_test[~mask].to(device), 1).squeeze().cpu()  # 外部区域

    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    bound = (grid[:, 0] == 1) | (grid[:, 1] == 1) | (grid[:, 0] == -1) | (grid[:, 1] == -1)
    # 边界采用ghost_cell
    x = grid[bound]
    col_point_i = collocation_point_sim(x)
    label_bound = torch.empty((col_point_i.size()[0], 5))

    u1_net.to('cpu').eval()
    u2_net.to('cpu').eval()
    for _ in range(5):
        rs = u1_net(col_point_i[:, _ * 2:_ * 2 + 2])
        label_bound[:, _] = rs.squeeze(1)
    # 计算PDE残差
    label_bound = bound_adjust(col_point_i, label_bound)
    pred[bound] = HCP_observe_pre(col_point_i.to(device).to(device), label_bound.to(device), 1).to('cpu').squeeze(-1).detach()  # 投影后的预测结果
    # 转换为NumPy数组
    X_np = xx.numpy()
    Y_np = yy.numpy()
    Z_pred = pred.reshape(n_points, n_points).numpy()
    Z_exact = exact.reshape(n_points, n_points).numpy()
    mask_np = mask.reshape(n_points, n_points).numpy()

    # 创建3D图形
    fig = plt.figure(figsize=(18, 6))

    # ----------------- 预测解3D图 -----------------
    ax1 = fig.add_subplot(131, projection='3d')

    # 分离内部和外部区域
    Z_pred_inside = np.where(mask_np, Z_pred, np.nan)
    Z_pred_outside = np.where(~mask_np, Z_pred, np.nan)

    # 绘制内部区域
    surf1 = ax1.plot_surface(X_np, Y_np, Z_pred_inside,
                             cmap='viridis', alpha=0.9, vmin=Z_pred.min(), vmax=Z_pred.max())

    # 绘制外部区域
    surf2 = ax1.plot_surface(X_np, Y_np, Z_pred_outside,
                             cmap='plasma', alpha=0.9, vmin=Z_pred.min(), vmax=Z_pred.max())

    # 添加界面轮廓线
    ax1.contour(X_np, Y_np, mask_np.astype(float), levels=[0.5],
                colors='gold', linewidths=2, linestyles='dashed')

    ax1.set_title('Predicted Solution', fontsize=12, pad=10)
    ax1.set_xlabel('x', fontsize=10, labelpad=8)
    ax1.set_ylabel('y', fontsize=10, labelpad=8)
    ax1.set_zlabel('u(x,y)', fontsize=10, labelpad=8)
    fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=10, pad=0.1, label='Solution Value')

    # ----------------- 精确解3D图 -----------------
    ax2 = fig.add_subplot(132, projection='3d')

    Z_exact_inside = np.where(mask_np, Z_exact, np.nan)
    Z_exact_outside = np.where(~mask_np, Z_exact, np.nan)

    surf3 = ax2.plot_surface(X_np, Y_np, Z_exact_inside,
                             cmap='viridis', alpha=0.9, vmin=Z_exact.min(), vmax=Z_exact.max())
    surf4 = ax2.plot_surface(X_np, Y_np, Z_exact_outside,
                             cmap='plasma', alpha=0.9, vmin=Z_exact.min(), vmax=Z_exact.max())
    ax2.contour(X_np, Y_np, mask_np.astype(float), levels=[0.5],
                colors='gold', linewidths=2, linestyles='dashed')

    ax2.set_title('Exact Solution', fontsize=12, pad=10)
    ax2.set_xlabel('x', fontsize=10, labelpad=8)
    ax2.set_ylabel('y', fontsize=10, labelpad=8)
    ax2.set_zlabel('u(x,y)', fontsize=10, labelpad=8)
    fig.colorbar(surf3, ax=ax2, shrink=0.6, aspect=10, pad=0.1, label='Solution Value')

    # ----------------- 误差2D图 -----------------

    error = np.abs(Z_pred - Z_exact)
    ax3 = fig.add_subplot(133)
    im = ax3.imshow(error,
                    cmap='viridis',
                    extent=[xx.min(), xx.max(), yy.min(), yy.max()],  # 设置坐标范围
                    origin='lower',  # 原点在左下角
                    aspect='auto')  # 自动调整纵横比

    # 添加颜色条
    fig.colorbar(im, ax=ax3, shrink=0.6, label='Error')
    ax3.set_title('Absolute Error')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.tight_layout()




    error = torch.abs(pred - exact)
    real_loss = torch.norm(pred - exact, 2)/ (torch.norm(exact, 2))
    print(f"Max error: {error.max().item():.4e}, Mean error: {error.mean().item():.4e}, L2_error : {real_loss.item():.4e}")


    # 调整视角和间距
    for ax in [ax1, ax2]:
        ax.view_init(elev=35, azim=45)
        ax.set_box_aspect([1, 1, 0.6])  # 调整z轴比例
        ax.grid(False)  # 隐藏网格线

    plt.tight_layout()
    plt.savefig('domain_HCP - 3.png')

    plt.show()


plot_results(u1_net, u2_net)