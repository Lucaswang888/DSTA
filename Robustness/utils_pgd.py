import torch
import torch.nn as nn
import torch.nn.functional as F

class PGD_DSTA:
    def __init__(self, model, eps=0.01, alpha=0.002, steps=10, random_start=True, device='cuda'):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.device = device

    def __call__(self, x, y, toa):
        # x: [batch, frames, obj, feat_dim]
        # y: [batch, 2]
        # toa: [batch]
        
        self.model.eval() # PGD生成期间保持eval模式（但在计算梯度）
        
        x_adv = x.clone().detach()

        if self.random_start:
            # 随机初始化扰动
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv, min=x.min(), max=x.max()) # 假设特征有范围，如果没有可以去掉或根据数据标准化调整

        for _ in range(self.steps):
            x_adv.requires_grad = True
            
            # Forward pass of DSTA
            # 注意：DSTA返回 (losses, all_outputs, all_hidden, all_alphas)
            losses, _, _, _ = self.model(x_adv, y, toa, nbatch=1, testing=False)
            
            # 我们攻击的目标是最大化 Total Loss (Cross Entropy + Aux)
            loss = losses['total_loss'].mean() # DSTA返回的是dict，需要根据你的模型逻辑取平均

            # Calculate gradients
            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]

            # Update adversarial images
            x_adv = x_adv.detach() + self.alpha * grad.sign()
            
            # Clip perturbation
            delta = torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
            x_adv = torch.clamp(x + delta, min=x.min(), max=x.max())

        return x_adv.detach()
