import torch
import random
import numpy as np

# 创建一个形状为[1, 11, 84]的张量
agents_obs = torch.randn(11, 10)
print("agent_obs",agents_obs)

# 随机生成要mask的行索引
radio = 4
mask_agent = random.sample(range(11), radio)
# 将指定行的数据全部设为0
agents_obs[mask_agent, :] = 0
print("agent_obs",agents_obs)