from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import random


# This multi-agent controller shares parameters between agents
class MASIAMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        self.mask_obs = []
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None
        self.encoder_hidden_states = None
        

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, teacher_forcing=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):#用来选择动作的，最后输出的是一个Q表格
        agent_inputs = self._build_inputs(ep_batch, t)#agent_inputs.shape torch.Size([11, 95]，95是84+11是因为后面11行是add了agent的id矩阵(11*11))
        if self.args.use_mask == True:
            agent_mask_inputs = self._build_mask_inputs(ep_batch, t)
        elif self.args.use_mask == False:
            agent_mask_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states, self.encoder_hidden_states = self.agent(agent_inputs, agent_mask_inputs, self.hidden_states, self.encoder_hidden_states)
        
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def rl_forward(self, ep_batch, state_repr, t, test_mode=False):#用来执行强化主干的
        # Go through downstream rl agent
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs, self.hidden_states = self.agent.rl_forward(agent_inputs, state_repr, self.hidden_states)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)        

    def enc_forward(self, ep_batch, t, test_mode=False):#用来执行强化训练时，观测通过encoder网络的
        agent_inputs = self._build_inputs(ep_batch, t)
        state_repr, self.encoder_hidden_states = self.agent.enc_forward(agent_inputs, self.encoder_hidden_states)
        if self.args.state_encoder in ["ob_attn_ae", "ob_attn_skipsum_ae", "ob_attn_skipcat_ae"]:
            return state_repr.view(ep_batch.batch_size, -1)
        else:
            return state_repr.view(ep_batch.batch_size, self.n_agents, -1)

    def vae_forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        if "vae" in self.args.state_encoder:
            recons, input, mu, log_var, self.encoder_hidden_states = self.agent.vae_forward(agent_inputs, self.encoder_hidden_states)
            return recons, input, mu, log_var
        elif "ae" in self.args.state_encoder:
            recons, input, z, self.encoder_hidden_states = self.agent.vae_forward(agent_inputs, self.encoder_hidden_states)
            return recons, input, z
        else:
            raise ValueError("Unsupported state encoder type!")
        
    def mask_vae_forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_mask_inputs(ep_batch, t)
        if "vae" in self.args.state_encoder:
            recons, input, mu, log_var, self.encoder_hidden_states = self.agent.vae_forward(agent_inputs, self.encoder_hidden_states)
            return recons, input, mu, log_var
        elif "ae" in self.args.state_encoder:
            recons, input, z, self.encoder_hidden_states = self.agent.vae_forward(agent_inputs, self.encoder_hidden_states)
            return recons, input, z
        else:
            raise ValueError("Unsupported state encoder type!")
    
    def target_transform(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        assert "vae" not in self.args.state_encoder, "Shouldn't use vae."
        # traget_projected.shape: [bs, spr_dim]
        target_projected, self.encoder_hidden_states = self.agent.target_transform(agent_inputs, self.encoder_hidden_states)
        return target_projected

    def init_hidden(self, batch_size, fat=False):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        # We share encoder_hidden_states between online encoder and target encoder
        if not fat:
            self.encoder_hidden_states = self.agent.encoder_init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)    # bav
        else:
            self.encoder_hidden_states = self.agent.encoder_init_hidden().unsqueeze(0).expand(batch_size*self.n_agents, self.n_agents, -1)    # bav

    def parameters(self):
        return self.agent.parameters()

    def rl_parameters(self):
        return self.agent.rl_parameters()

    def enc_parameters(self):
        return self.agent.enc_parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs
    

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape


    def _build_mask_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        mask_inputs = batch["mask_obs"][:, t] 
        inputs.append(mask_inputs)
        # print("mask_input_shape",len(mask_inputs),",",len(mask_inputs[0]),",",len(mask_inputs[0][0]))
        # print("mask input",mask_inputs)
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs
        
        
    # def _build_supplement_inputs(self, batch, t):
    #     bs = batch.batch_size
    #     inputs = []
    #     if self.args.obs_use_sequence_sup:
    #         k = 1
    #         mask_inputs = self._mask_input(batch["obs"][:, t])
    #         sup_matrix = th.zeros_like(batch["obs"][:, t])
    #         if t>=k:
    #             #obs_matrix = batch["obs"][:, t]
    #             replace_matrix = batch["obs"][:, t-k]
    #             sup_matrix = th.where((replace_matrix == 0),replace_matrix, mask_inputs)
    #         inputs.append(sup_matrix)  # b1av batch["obs"][:, t].shape = [1,11,84]
    #     if self.args.obs_last_action:
    #         if t == 0:
    #             inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
    #         else:
    #             inputs.append(batch["actions_onehot"][:, t-1])
    #     if self.args.obs_agent_id:
    #         inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
    #     inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
    #     return inputs
    
    # def _mask_input(self, input_tensor):
    #     # 获取输入张量的形状
    #     mask_matrix = input_tensor
    #     # 随机生成要mask的行索引
    #     num_rows_to_mask = self.args.mask_num  # 你可以根据需要调整要mask的行数
    #     rows_to_mask = th.randint(0, 11, (num_rows_to_mask,))
    #     # 将指定行的数据全部设为0
    #     mask_matrix[:, rows_to_mask, :] = 0
    #     return mask_matrix
    
    def _get_mask_obs_dimension(self, batch, t):
        bs = batch.batch_size
        obs = batch["obs"][:, t]
        dimension = len(obs[0][0]) 
        ratio = self.args.ratio
        mask_num = int(ratio*dimension)
        # 将指定行的数据全部设为0
        for i in range(bs):
            for j in range(len(obs[0])):
                mask_indices = random.sample(range(dimension), mask_num)
                for idx in mask_indices:
                    obs[i][j][idx] = 0
        return obs
        
    def _get_mask_obs_agent(self, batch, t):
        bs = batch.batch_size
        obs = batch["obs"][:, t]
        agentnum = len(obs[0])
        ratio = self.args.ratio
        mask_num = int(ratio*agentnum)
        # print("mask_num",mask_num)
        mask_agent = random.sample(range(agentnum), mask_num)
        # print("mask_agent",mask_agent)
        # 将指定行的数据全部设为0
        for i in range(bs):
            for row in mask_agent:
                obs[i][row] = th.zeros_like(obs[i][row])
        return obs
    
    def _get_mask_obs(self, batch, t):
        if self.args.mask_method == "dimension":
            return self._get_mask_obs_dimension(batch,t)
        if self.args.mask_method == "agent":
            return self._get_mask_obs_agent(batch,t)   