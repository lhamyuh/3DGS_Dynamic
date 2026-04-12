import torch
import torch.nn as nn

class DeformModel(nn.Module):
    def __init__(self, D=4, W=256):
        super().__init__()
        # 5090 显存很大，我们可以把 W 设为 256 提高精度
        self.net = nn.Sequential(
            nn.Linear(4, W), nn.ReLU(),
            nn.Linear(W, W), nn.ReLU(),
            nn.Linear(W, W), nn.ReLU(),
            nn.Linear(W, 3) # 输出 xyz 的偏移量
        )
        # 初始化为 0，保证训练开始时高斯点在原位，不会“飞走”
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, t):
        # 确保 t 无论传入的是数字还是单元素 Tensor，都能转为标量
        if isinstance(t, torch.Tensor):
            t = t.item()

        # x: [N, 3], t: float
        t_tensor = torch.full((x.shape[0], 1), t).to(x.device)
        inp = torch.cat([x, t_tensor], dim=-1)
        return self.net(inp)