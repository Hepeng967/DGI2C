import torch

# 假设你的张量为 tensor，大小为 (20, 11, 84)
tensor = torch.rand(20, 11, 84)

# 创建一个大小为 (11, 11) 的单位矩阵
identity_matrix = torch.eye(11)

# 在最后两个维度上拼接单位矩阵
tensor_with_identity = torch.cat([tensor, identity_matrix.unsqueeze(0).expand(20, -1, -1)], dim=-1)

# 打印结果
print(tensor_with_identity.shape)
