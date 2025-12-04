import os
import torch
import torch.nn as nn
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

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
        res = torch.tensor(-4.0)
    return res

def g(x,y):
    r = torch.sqrt(x**2 + y**2)
    res = (r**4 + 0.1 * torch.log(2 * r)) / alpha1
    return res


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
    r = 0.5 + r1 * torch.sin(OMEGA * theta)   # 极坐标下的界面方程
    x_gamma = r * torch.cos(theta)
    y_gamma = r * torch.sin(theta)
    samples['X_gamma'] = torch.cat([x_gamma, y_gamma], dim=1).to(device)



    # 内部区域采样
    theta = torch.rand(N_inside, 1) * 2 * torch.pi
    r_max = 0.5 + r1 * torch.sin(OMEGA * theta)
    r = torch.sqrt(torch.rand(N_inside, 1)) * r_max
    samples['X_inside2'] = torch.cat([r * torch.cos(theta), r * torch.sin(theta)], dim=1).to(device)

    # 外部区域采样
    theta = torch.rand(N_outside, 1) * 2 * torch.pi
    r_min = 0.5 + r1 * torch.sin(OMEGA * theta)   # 太阳花边界
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




# 2. 定义神经网络结构
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


rho = 0.5
beta = 0.01
max_outer_iter = 1
inner_epochs = 60000
N_omega = 100
N_gamma = 100
N_inside1 = 500
N_inside2 = 500



samples = generate_samples(
    N_omega=N_omega,
    N_gamma=N_gamma,
    N_outside=N_inside1,
    N_inside=N_inside2,
    device=device
)

observe_ = generate_samples(
    N_omega=100,
    N_gamma=100,
    N_inside=100,
    N_outside=100,
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
loss_old = 10000

phi = torch.zeros(N_gamma, 1)
Phi = torch.zeros(N_gamma, 1)



##################################################################################

seed = 50
deltaX = 0.01
deltaY = 0.01


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
    u1 = u_net1(X)
    u2 = u_net2(X)
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


def compute_boundary_loss(u_net, X):
    u = u_net(X)
    return torch.mean((u - g(X[:, 0:1], X[:, 1:2]))**2)

def obs_loss(u_net, X):
    u = u_net(X)
    realdata = exact_solution(X[:, 0:1], X[:, 1:2])
    return torch.mean((u - realdata) ** 2)



def compute_loss(u1_net, u2_net, X_inside, X_d, X_gamma, phi, Phi , area):
    # 启用梯度计算
    X_inside.requires_grad_(True)
    X_d = X_d.to(device)
    X_omega  = X_inside.to(device)
    X_gamma = X_gamma.to(device)

    if area:
        u = u1_net(X_omega)

        grad_u = torch.autograd.grad(u.sum(), X_omega, create_graph=True)[0]
        grad_u_x, grad_u_y = grad_u[:, 0:1], grad_u[:, 1:2]

        grad_u_x2 = torch.autograd.grad(grad_u_x.sum(), X_omega, create_graph=True)[0][:, 0:1]
        grad_u_y2 = torch.autograd.grad(grad_u_y.sum(), X_omega, create_graph=True)[0][:, 1:2]
        delta_u = grad_u_x2 + grad_u_y2

        F = f(X_omega[:, 0:1], X_omega[:, 1:2], area)
        pde_loss = torch.mean((alpha1 * delta_u + F) ** 2)

        # 边界损失
        d_loss = compute_boundary_loss(u1_net, X_d)

        interface_loss = compute_interface_loss(u1_net,u2_net, X_gamma, phi, Phi, 1)

        # # 观测损失
        obs_gamma_loss = obs_loss(u1_net, train_point.N_inside1)

        total_loss = pde_loss + 20 * obs_gamma_loss + 40 * d_loss + beta * interface_loss


    else:

        u = u2_net(X_inside)
        grad_u = torch.autograd.grad(u.sum(), X_inside, create_graph=True)[0]
        grad_u_x, grad_u_y = grad_u[:, 0:1], grad_u[:, 1:2]

        grad_u_x2 = torch.autograd.grad(grad_u_x.sum(), X_inside, create_graph=True)[0][:, 0:1]
        grad_u_y2 = torch.autograd.grad(grad_u_y.sum(), X_inside, create_graph=True)[0][:, 1:2]
        delta_u = grad_u_x2 + grad_u_y2

        F = f(X_inside[:, 0:1], X_inside[:, 1:2], area)
        pde_loss = torch.mean((alpha2 * delta_u + F) ** 2)


        interface_loss = compute_interface_loss(u1_net,u2_net, X_gamma, phi, Phi, 0)

        # 内部观测损失
        obs_gamma_loss = obs_loss(u2_net, train_point.N_inside2)

        # 界面损失
        d_loss = obs_loss(u2_net, X_gamma)

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



minloss = 1e10
minloss2 = 1e10

B = torch.eye(3)
B[0][0] = 0

for k in range(max_outer_iter):
    print(f"\nOuter Iteration {k + 1}/{max_outer_iter}")


    # 训练子问题1
    for epoch in range(inner_epochs):
        optimizer1.zero_grad()
        total_loss1, pdeloss1, interfaceloss1 = compute_loss(u1_net, u2_net, X_inside1,X_omega, X_gamma, phi, Phi, 1)
        _, l2, _ = compute_loss(u1_net, u2_net, X_inside2, X_omega, X_gamma, phi, Phi, 1)
        total_loss1 += l2
        total_loss1 = total_loss1
        total_loss1.backward()
        optimizer1.step()
        if epoch % 100 == 0:
            print(f"Subproblem 1 - Epoch {epoch}: Loss = {total_loss1.item():.4e},pdeLoss = {pdeloss1.item():.4e}, interfaceLoss = {interfaceloss1.item():.4e}")

    realdata1 = exact_solution(X_inside1[:, 0].detach().to('cpu'), X_inside1[:, 1].detach().to('cpu'))
    batch_size = X_inside1.shape[0]
    B_batch = B.unsqueeze(0).expand(batch_size, -1, -1).to(device)
    u1 = u1_net(X_inside1)
    real_loss1 = torch.norm(u1[:, 0] - realdata1.clone().detach().to(device), 2)/ (torch.norm( realdata1, 2))

    if total_loss1 < minloss:
        minloss = total_loss1
        best_model1 = copy.deepcopy(u1_net.state_dict())


    # 更新接口条件
    with torch.no_grad():
        u1_gamma = u1_net(X_gamma)
        u2_gamma = u2_net(X_gamma)
        phi = rho * u2_gamma + (1 -rho) * u1_gamma

    # 训练子问题2
    for epoch in range(inner_epochs):
        optimizer2.zero_grad()
        total_loss2, pdeloss2, interfaceloss2 = compute_loss(u1_net, u2_net, X_inside2,X_omega, X_gamma, phi, Phi, 0)
        total_loss2.backward()
        optimizer2.step()
        if epoch % 100 == 0:
            print(f"Subproblem 2 - Epoch {epoch}: Loss = {total_loss2.item():.4e}, pdeLoss = {pdeloss2.item():.4e}, interfaceLoss = {interfaceloss2.item():.4e}")

    realdata2 = exact_solution(X_inside2[:, 0].detach().to('cpu'), X_inside2[:, 1].detach().to('cpu'))
    u2 = u2_net(X_inside2)
    real_loss2 = torch.norm(u2[:, 0] - realdata2.clone().detach().to(device), 2)/ (torch.norm(realdata2, 2))

    if total_loss2 < minloss2:
        minloss2 = total_loss2
        best_model2 = copy.deepcopy(u2_net.state_dict())

    # 交换Dirichlet迹
    with torch.no_grad():  # 全局禁用梯度
        # 局部启用梯度计算
        with torch.enable_grad():
            u2_gamma = u2_net(X_gamma)
            u2_grad = torch.autograd.grad(u2_gamma[:, 0].sum(), X_gamma, create_graph=True)[0]

        theta = torch.atan2(X_gamma[:, 0], X_gamma[:, 1])
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

        Phi = - alpha2 * (u2_grad[:, 0:1] * n_x + u2_grad[:, 1:2] * n_y)

    u1_net.load_state_dict(best_model1)
    u2_net.load_state_dict(best_model2)


    u1 = u1_net(X_inside1)
    u2 = u2_net(X_inside2)

    real_loss1 = torch.norm((u1[:, 0] - realdata1.clone().detach().to(device))
    , 2)/ (torch.norm((realdata1), 2))

    real_loss2 = torch.norm((u2[:, 0] - realdata2.clone().detach().to(device))
    , 2)/ (torch.norm((realdata2), 2))


    print(f'realloss1 = {real_loss1}, realloss2 = {real_loss2}')


def plot_results(u1_net, u2_net):
    # 将模型移到CPU并设置为评估模式
    u1_net.to('cpu').eval()
    u2_net.to('cpu').eval()

    n_points = 200  # 网格分辨率

    # 生成测试网格 [-1,1]×[-1,1]
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
        pred[mask] = u2_net(X_test[mask]).squeeze()  # 内部区域
        exact = exact_solution(X_test[:, 0], X_test[:, 1])
        pred[~mask] = u1_net(X_test[~mask]).squeeze()  # 外部区域
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

    # 绘制内部区域（冷色调）
    surf1 = ax1.plot_surface(X_np, Y_np, Z_pred_inside,
                             cmap='viridis', alpha=0.9, vmin=Z_pred.min(), vmax=Z_pred.max())

    # 绘制外部区域（暖色调）
    surf2 = ax1.plot_surface(X_np, Y_np, Z_pred_outside,
                             cmap='plasma', alpha=0.9, vmin=Z_pred.min(), vmax=Z_pred.max())

    # 添加界面轮廓线（金色更醒目）
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
    plt.savefig('pretrain3.png')
    plt.show()
    plt.close()


plot_results(u1_net, u2_net)