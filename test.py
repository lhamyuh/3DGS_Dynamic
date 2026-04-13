import torch
from model.deform_model import DeformModel

# 1. 模拟输入数据
x = torch.randn(10, 3).cuda()
t = torch.tensor([0.5]).cuda()

# 2. 实例化模型
# 注意：input_ch(39) + input_ch_time(21) = 60
try:
    model = DeformModel(D=8, W=256, input_ch=39, input_ch_time=21).cuda()

    # 3. 前向传播测试
    out = model(x, t)

    print("-" * 30)
    print(f"✅ V7 维度校验成功！")
    print(f"   输入空间维度: {model.input_ch} (PE 6阶)")
    print(f"   输入时间维度: {model.input_ch_time} (PE 10阶)")
    print(f"   总输入维度: {model.input_ch + model.input_ch_time}")
    print(f"   输出位移形状: {out.shape} (应为 [10, 3])")
    print("-" * 30)

except Exception as e:
    print("-" * 30)
    print(f"❌ 维度匹配失败！错误信息:")
    print(e)
    print("-" * 30)