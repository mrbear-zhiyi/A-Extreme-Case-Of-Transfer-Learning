import torch
import torch.nn as nn
"""
神经网络结构模块

=====网络结构====
1、DNN结构

=====求解器配置====
1、LM求解器
"""



"""
================================================DNN结构================================================
"""
class AlphaDecayNN(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: int, hidden_neurons: int):
        super(AlphaDecayNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_neurons))
        layers.append(nn.ReLU())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_neurons, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).squeeze()



"""
================================================求解器配置================================================
"""
class AdaptiveLevenbergMarquardtOptimizer:

    def __init__(self, model, lambda_=None, max_iter=100, tol=1e-6, device='cpu'):
        self.model = model
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.lambda_ = lambda_ if lambda_ is not None else 1e-3
        self.min_lambda = 1e-12
        self.max_lambda = 1e8
        self.max_delta_norm = 1.0
        self.trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.num_trainable_params = sum(p.numel() for p in self.trainable_params)

        print(f"LM Optimizer initialized with {self.num_trainable_params} trainable parameters "
              f"out of {sum(p.numel() for p in model.parameters())} total parameters")



    # ==============================================计算雅可比行列式==================================================
    def compute_jacobian(self, x):
        batch_size = x.shape[0]


        # 无训练参数
        if self.num_trainable_params == 0:
            print("Warning: No trainable parameters. Returning zero Jacobian.")
            return torch.zeros(batch_size, 0, device=self.device), torch.zeros(batch_size, 1, device=self.device)
        # 初始化Jacobian矩阵
        jacobian = torch.zeros(batch_size, self.num_trainable_params, device=self.device)
        outputs_list = []


        # 单个样本单独计算
        for i in range(batch_size):
            x_i = x[i:i + 1].detach().clone().requires_grad_(True)
            self.model.zero_grad()
            output_i = self.model(x_i)
            outputs_list.append(output_i.item())
            # 计算可训练参数梯度
            if output_i.requires_grad:
                grad_outputs = torch.ones_like(output_i)
                try:
                    grads = torch.autograd.grad(
                        output_i,
                        self.trainable_params,
                        grad_outputs=grad_outputs,
                        create_graph=False,
                        allow_unused=True
                    )
                    # 拼接可训练参数梯度
                    grad_flat = []
                    for g, p in zip(grads, self.trainable_params):
                        if g is not None:
                            grad_flat.append(g.view(-1))
                        else:
                            grad_flat.append(torch.zeros_like(p).view(-1))

                    if grad_flat:
                        jacobian[i] = torch.cat(grad_flat)


                except Exception as e:
                    print(f"Gradient computation failed for sample {i}: {e}")
                    jacobian[i] = torch.zeros(self.num_trainable_params, device=self.device)

        outputs = torch.tensor(outputs_list, device=self.device).view(batch_size, 1)
        return jacobian, outputs


    # ==============================================前向传播==================================================
    def step(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)


        # 计算损失
        if y.dim() == 1:
            y = y.unsqueeze(1)
        J, outputs = self.compute_jacobian(x)
        residuals = outputs - y
        current_loss = torch.mean(residuals ** 2).item()
        # 无训练参数
        if self.num_trainable_params == 0:
            print("Warning: No trainable parameters. Skipping optimization step.")
            return current_loss
        # 构建LM矩阵
        JTJ = torch.matmul(J.t(), J)
        JTr = torch.matmul(J.t(), residuals)
        # 添加阻尼项
        damping = self.lambda_ * torch.eye(JTJ.shape[0], device=self.device)


        try:
            # 求解线性系统
            delta = torch.linalg.solve(JTJ + damping, -JTr)
            # 控制步长
            delta_norm = torch.norm(delta)
            if delta_norm > self.max_delta_norm:
                delta = delta * (self.max_delta_norm / delta_norm)
            # 更新参数
            start = 0
            for param in self.trainable_params:
                end = start + param.numel()
                param_update = delta[start:end].view(param.shape)
                param.data = param.data + param_update
                start = end
            # 评估新损失
            with torch.no_grad():
                new_outputs = self.model(x)
                if new_outputs.dim() == 1:
                    new_outputs = new_outputs.unsqueeze(1)
                new_residuals = new_outputs - y
                new_loss = torch.mean(new_residuals ** 2).item()
            # 更新策略
            if new_loss < current_loss:
                self.lambda_ = max(self.lambda_ / 10, self.min_lambda)
                return new_loss
            else:
                self.lambda_ = min(self.lambda_ * 10, self.max_lambda)
                # 回滚参数
                start = 0
                for param in self.trainable_params:
                    end = start + param.numel()
                    param.data = param.data - delta[start:end].view(param.shape)
                    start = end
                return current_loss

        except Exception as e:
            print(f"LM step failed: {e}")
            self.lambda_ = min(self.lambda_ * 10, self.max_lambda)
            return current_loss