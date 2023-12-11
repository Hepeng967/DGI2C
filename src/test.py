import numpy as np
import random
bs = 1
array_3d = [[[1 for _ in range(10)] for _ in range(10)]]
obs = array_3d
# 打印数组
print(array_3d)
agentnum = len(obs[0])
print(agentnum)
ratio = 0.2
dimension = len(obs[0][0])
# 将指定行的数据全部设为0
print(len(obs),len(obs[0]),len(obs[0][0]))
print(len(obs[0]))
print(len(obs[0][0]))
mask_dimension = int(ratio*dimension)
print("mask_dimension",mask_dimension)
# 将指定行的数据全部设为0
for i in range(bs):
    for j in range(len(obs[0])):
        mask_indices = random.sample(range(dimension), mask_dimension)
        for idx in mask_indices:
            obs[i][j][idx] = 0
print ("obs",obs)