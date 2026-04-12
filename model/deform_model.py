import torch
import torch.nn as nn
import math


class DeformModel(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_time=21):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = [4]  # 在第 4 层加入残差连接，增强深层网络的梯度传导

        # 第一层：输入维度 = 3 (xyz) + 21 (encoded t) = 24
        self.linears = nn.ModuleList(
            [nn.Linear(input_ch + input_ch_time, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch + input_ch_time, W) for i in
             range(D - 1)]
        )

        # 输出层：输出位移 delta_xyz
        self.output_linear = nn.Linear(W, 3)

        # 最后一层初始化为 0，确保训练开始时高斯点在原位（Canonical Space）
        nn.init.zeros_(self.output_linear.weight)
        nn.init.zeros_(self.output_linear.bias)

    def forward(self, x, t):
        # 1. 确保 t 的形状为 [N, 1]
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=x.device, dtype=x.dtype)
        if t.dim() == 0: t = t.view(1, 1)
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], 1)

        # 2. --- 【时间位置编码 (Temporal PE)】 ---
        t_emb = [t]
        L = 10  # 10 阶编码
        for i in range(L):
            for fn in [torch.sin, torch.cos]:
                t_emb.append(fn(2.0 ** i * math.pi * t))
        t_encoded = torch.cat(t_emb, dim=-1)  # [N, 21]

        # 3. 拼接输入：[N, 3] + [N, 21] -> [N, 24]
        h = torch.cat([x, t_encoded], dim=-1)
        input_pts = h

        # 4. 前向传播（带 Skip Connection）
        for i, l in enumerate(self.linears):
            h = self.linears[i](h)
            h = torch.nn.functional.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], dim=-1)

        return self.output_linear(h)