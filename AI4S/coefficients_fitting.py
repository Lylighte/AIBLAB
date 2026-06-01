import pandas as pd
import numpy as np
from numpy.typing import NDArray as ND
import torch
from torch.utils import data
from torch import nn
from timer import Timer


# 进行拟合系数的函数
def fit(x: pd.DataFrame, y: ND, method='analytical',
        num_epochs=40, batch_size=100, lr=0.1):
    '''
    系数拟合(只考虑一维情况)
    method: 'analytical' 解析解, 'nn' 神经网络解
    '''
    # 将pd数据转为tensor (CPU)
    X = torch.tensor(x.to_numpy())
    y = torch.tensor(y)

    # 检查数据形状，若不是列向量则转为列向量
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # 创造储存结果的数组
    r2_out = [None] * (X.shape[1])
    p_out = [None] * (X.shape[1])
    loss_out = [None] * (X.shape[1])
    time_out = [None] * (X.shape[1])

    # 计时器
    timer_ana = Timer()
    timer_nn = Timer()

    # X的每列数据即为y=a*f(x1,x2,..)+b中的f(x1,x2,..)方程数据
    # 对每列数据分别拟合系数
    for i in range(X.shape[1]):
        if method == 'nn':
            timer_nn.start()
            r2, p, loss = net_solving(X[:, i], y[:],
                                       num_epochs=num_epochs,
                                       batch_size=batch_size,
                                       lr=lr)
            t = timer_nn.stop()
            time_out[i] = t
        else:
            timer_ana.start()
            r2, p, loss = analytical_solving(X[:, i], y[:])
            t = timer_ana.stop()
            time_out[i] = t

        # 保存结果
        r2_out[i] = r2
        p_out[i] = p
        loss_out[i] = loss

    # 输出累计耗时
    if method == 'nn':
        print(f"[神经网络] 总耗时: {timer_nn.sum():.4f}s, "
              f"平均: {timer_nn.avg():.4f}s/特征, "
              f"处理特征数: {X.shape[1]}")
    else:
        print(f"[解析解] 总耗时: {timer_ana.sum():.4f}s, "
              f"平均: {timer_ana.avg():.4f}s/特征, "
              f"处理特征数: {X.shape[1]}")

    return r2_out, p_out, loss_out, time_out


def get_loss(y_true, y_pred):
    '''
    计算损失函数
    '''
    return float(torch.mean((y_true - y_pred) ** 2) / 2)


def get_r2(y_true, y_pred):
    '''
    计算R2
    '''
    y_true_mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - y_true_mean) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    return float(1 - (ss_residual / ss_total))


def analytical_solving(X: torch.tensor, y: torch.tensor, lambda_reg=1e-6):
    '''
    解析求解线性回归系数
    lambda_reg: 正则化系数（防止 X^T X 不可逆）
    '''
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # 增加一列全为1的列，用于计算截距 b
    ones = torch.ones((X.shape[0], 1))
    X = torch.hstack((X, ones))

    # 解析解: c = (X^T X)^-1 X^T y
    X_T_X = X.T @ X
    X_T_X_reg = X_T_X + lambda_reg * torch.eye(X_T_X.shape[0], dtype=torch.float64)
    X_T_y = X.T @ y
    XTX_inv = torch.linalg.inv(X_T_X_reg)
    p = XTX_inv @ X_T_y

    r2 = get_r2(y, X @ p)
    loss = get_loss(y, X @ p)

    p = p.flatten().tolist()
    return r2, p, loss


def net_solving(X: torch.tensor, y: torch.tensor,
                num_epochs=40, batch_size=100, lr=0.1):
    '''
    神经网络求解线性回归系数 (单层 Linear(1,1))
    '''
    def load_array(data_arrays, batch_size, is_train=False):
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)

    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    features = X.float()
    labels = y.float()

    data_iter = load_array((features, labels), batch_size)

    # 单层线性模型
    net = nn.Sequential(nn.Linear(1, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    loss_fn = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for X_iter, y_iter in data_iter:
            l = loss_fn(net(X_iter), y_iter)
            trainer.zero_grad()
            l.backward()
            trainer.step()

    # 提取参数
    p = np.array([])
    for w in net[0].weight.data.numpy():
        p = np.append(p, w)
    p = np.append(p, net[0].bias.data.numpy()[0])
    p = p.reshape(-1, 1)

    # 计算R2
    X_np = X.numpy()
    X_np = np.hstack((X_np, np.ones((X_np.shape[0], 1))))
    r2 = get_r2(y, X_np @ p)

    p = [item for sublist in p for item in sublist]
    return r2, p, float(loss_fn(net(features), labels))