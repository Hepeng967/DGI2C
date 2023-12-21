
import random

# obs = [[random.randint(0,9) for _ in range(6)] for _ in range(6)] 
# print("obs",obs)
# dimension = len(obs[0]) 
# ratio = 0.5
# mask_num = int(ratio*dimension)
# # 将指定行的数据全部设为0
# for i in range(len(obs)-1):
#     mask_indices = random.sample(range(dimension), mask_num)
#     middle = obs[i]
#     for j in mask_indices:
#         middle[j]=0
#     obs[i] = middle
# print("obs_next",obs)

# for i in range(1,5):
#     print(i)
# for i in range(5):
#     print(i)
obs = [[random.randint(0,9) for _ in range(6)] for _ in range(6)]    
agentnum = 7
ratio = 0.5
mask_num = int(ratio*(agentnum-1))
mask_num = int(ratio*agentnum)
# print("mask_num",mask_num)
mask_agent = random.sample(range(agentnum-1), mask_num)
print(mask_agent)
mask_agent = random.sample(range(1,agentnum), mask_num)
print(mask_agent)