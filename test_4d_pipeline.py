import torch
from model.deform_model import DeformModel
from gaussian_renderer import render
import os


# 1. 模拟一个简单的 4D 渲染流程
def check_pipeline():
    print("🚀 开始 4D 渲染管线测试...")

    # 模拟高斯点坐标 (假设有 1000 个点)
    means3D = torch.randn((1000, 3), device="cuda", requires_grad=True)
    time = 0.5

    # 初始化你的模型
    deform_model = DeformModel(D=4, W=256).cuda()

    # 前向传播：计算位移
    d_xyz = deform_model(means3D, time)
    means3D_deformed = means3D + d_xyz

    print(f"✅ 前向传播成功! 平均位移幅度: {d_xyz.abs().mean().item():.6f}")

    # 模拟一个 Loss (比如让位移越大越好，纯粹为了测试梯度)
    loss = d_xyz.sum()
    loss.backward()

    # 检查梯度：如果 grad 不是 None，说明变形场已成功挂钩
    has_grad = False
    for name, param in deform_model.named_parameters():
        if param.grad is not None:
            has_grad = True
            break

    if has_grad:
        print("✅ 梯度校验成功！DeformModel 已正确接入计算图。")
    else:
        print("❌ 错误：DeformModel 没有接收到梯度，请检查 forward 逻辑。")


if __name__ == "__main__":
    check_pipeline()