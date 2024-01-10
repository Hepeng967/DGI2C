import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer\

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.optim import Adam
from components.standarize_stream import RunningMeanStd
import os
import time

class DGI2CLearner:
    def __init__(self, mac, latent_model, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.latent_model = latent_model
        self.logger = logger

        if not self.args.rl_signal:
            assert 0, "Must use rl signal in this method !!!"
            self.params = list(mac.rl_parameters())
        else:
            self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        if self.args.use_latent_model:
            # use_latent_model means use_spr
            self.params += list(latent_model.parameters())

        self.optimiser = Adam(params=self.params, lr=args.lr)
                
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

    def repr_train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # states.shape: [batch_size, seq_len, state_dim]
        states = batch["state"]
        # actions.shape: [batch_size, seq_len, n_agents, 1]
        actions_onehot = batch["actions_onehot"]
        actions = batch["actions"]
        rewards = batch["reward"]
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        # go through vae
        recons, z = [], []
        self.mac.init_hidden(batch.batch_size)  # useless in current version
        for t in range(batch.max_seq_length):#TODO 这个地方的max_seq_length相当于cogfig文件里的episode_limit + 1是不是
            recons_t, _, z_t = self.mac.vae_forward(batch, t) #vae_forward就是过一下encoder，中间那个返回的是input，我们不需要
            recons.append(recons_t)
            z.append(z_t)
        # recons.shape: [batch_size, seq_len, state_repre_dim]
        recons = th.stack(recons, dim=1)  # Concat over time
        z = th.stack(z, dim=1) #此时Z存的是（Z1到Zt的序列，recons表示重建的序列s^t）
        
        mask_recons, mask_z = [], []
        self.mac.init_hidden(batch.batch_size)  
        for t in range(batch.max_seq_length):
            mask_recons_t, _,mask_z_t = self.mac.mask_vae_forward(batch, t) 
            mask_recons.append(mask_recons_t)
            mask_z.append(mask_z_t)
            # if t==(batch.max_seq_length-1):
            #     print("mask_recons_t",mask_recons_t.shape)
        # recons.shape: [batch_size, seq_len, state_repre_dim]
        # print("mask_recons_before_stack",mask_recons.shape)
        mask_recons = th.stack(mask_recons, dim=1)  # Concat over time
        # print("mask_recons_after_stack",mask_recons.shape)
        mask_z = th.stack(mask_z, dim=1)
        

        
        bs, seq_len  = states.shape[0], states.shape[1]
        #loss_dict = self.mac.agent.encoder.loss_function(recons.reshape(bs*seq_len, -1), states.reshape(bs*seq_len, -1))#返回的是{loss：||s^t - st||**2}
        if self.args.use_mask == True:
            # print("mask_recons_after_reshape",(mask_recons.reshape(bs*seq_len, -1)).shape)
            # print("state_after_reshape",(mask_recons.reshape(bs*seq_len, -1)).shape)
            loss_dict = self.mac.agent.encoder.loss_function(mask_recons.reshape(bs*seq_len, -1), states.reshape(bs*seq_len, -1))#用mask和recons去计算mae
            # print("mask_recons")
        elif self.args.use_mask == False:
            loss_dict = self.mac.agent.encoder.loss_function(recons.reshape(bs*seq_len, -1), states.reshape(bs*seq_len, -1))
        vae_loss = loss_dict["loss"].reshape(bs, seq_len, 1)
        mask = mask.expand_as(vae_loss)
        vae_loss = (vae_loss * mask).sum() / mask.sum()#TODO 其实还是不太懂这个地方mask的意义，感觉像是VAE自带的

        if self.args.use_latent_model:
            # Compute target z first
            target_projected = []
            with th.no_grad():
                self.mac.init_hidden(batch.batch_size)#初始化网络的权重w
                for t in range(batch.max_seq_length):
                    target_projected_t = self.mac.target_transform(batch, t)
                    target_projected.append(target_projected_t)
            target_projected = th.stack(target_projected, dim=1)  # Concat over time, shape: [bs, seq_len, spr_dim]

            curr_z = z
            # Do final vector prediction*******************************************************
            predicted_f = self.mac.agent.online_projection(curr_z)   # [bs, seq_len, spr_dim]#通过 self.projection,和 self.final_classifier所得到的一个表示
            tot_spr_loss = self.compute_spr_loss(predicted_f, target_projected, mask)#compute_spr_loss相当于就是把mask引入后把前两项做MSE误差计算
            # if  self.args.use_inverse_model:
            #     predicted_act = F.softmax(self.latent_model.predict_action(z[:,:-1],z[:,1:]),dim=-1)
            #     predicted_act = predicted_act.reshape(*predicted_act.shape[:-2], -1)
            #     sample_act = actions_onehot[:,:-1].reshape(*actions_onehot[:,:-1].shape[:-2], -1)
            #     tot_inv_loss = self.compute_inv_loss(predicted_act, sample_act, mask[:,:-1])
            if  self.args.use_rew_pred:
                predicted_rew = self.latent_model.predict_reward(curr_z)   # [bs, seq_len, 1]
                tot_rew_loss = self.compute_rew_loss(predicted_rew, rewards, mask)
            for t in range(self.args.pred_len):#这个pred_len是定义的起始的k，t+k-1就等于
                original_z = curr_z
                # do transition model forward
                # curr_z_inv = self.latent_model(curr_z, actions_onehot[:, t:])
                curr_z = self.latent_model(curr_z, actions_onehot[:, t:])[:, :-1] #调用transition model mlp 
                
                # Do final vector prediction
                predicted_f = self.mac.agent.online_projection(curr_z)  # [bs, seq_len, spr_dim] 这个是只包含t+1~t+K的
                tot_spr_loss += self.compute_spr_loss(predicted_f, target_projected[:, t+1:], mask[:, t+1:])#target_projected是包含1~t+K的

                # if self.args.use_inverse_model:
                #     predicted_act = F.softmax(self.latent_model.predict_action(z[:,t:-1],z[:,t+1:]),dim=-1)
                #     predicted_act = predicted_act.reshape(*predicted_act.shape[:-2], -1)
                #     sample_act = actions_onehot[:,t:-1].reshape(*actions_onehot[:,t:-1].shape[:-2], -1)
                #     tot_inv_loss += self.compute_inv_loss(predicted_act, sample_act, mask[:,t:-1])
                if self.args.use_rew_pred:
                    predicted_rew = self.latent_model.predict_reward(curr_z)
                    tot_rew_loss += self.compute_rew_loss(predicted_rew, rewards[:, t+1:], mask[:, t+1:])
            #这个地方的coef是yaml里面配置的，用来配置各个损失之间的权重
            if self.args.use_rew_pred:
                repr_loss = vae_loss + self.args.spr_coef * tot_spr_loss + self.args.rew_pred_coef * tot_rew_loss
            elif self.args.dont_use_latent_loss:
                repr_loss = vae_loss
            else:
                # repr_loss = vae_loss + self.args.spr_coef * tot_spr_loss
                repr_loss = vae_loss + self.args.spr_coef * tot_spr_loss
        else:
            repr_loss = vae_loss
            
        if  self.args.use_inverse_model:
            predicted_act = F.softmax(self.latent_model.predict_action(z[:,:-1],z[:,1:]),dim=-1)
            predicted_act = predicted_act.reshape(*predicted_act.shape[:-2], -1)
            sample_act = actions_onehot[:,:-1].reshape(*actions_onehot[:,:-1].shape[:-2], -1)
            tot_inv_loss = self.compute_inv_loss(predicted_act, sample_act, mask[:,:-1])
            repr_loss += tot_inv_loss
        
        if t_env % self.args.learner_log_interval == 0:
            self.logger.log_stat("repr_loss", repr_loss.item(), t_env)
            self.logger.log_stat("vae_loss", vae_loss.item(), t_env)
            if self.args.use_latent_model:
                self.logger.log_stat("model_loss", tot_spr_loss.item(), t_env)
                if self.args.use_rew_pred:
                    self.logger.log_stat("rew_pred_loss", tot_rew_loss.item(), t_env)
            if self.args.use_inverse_model:
                self.logger.log_stat("inverse_model_loss",tot_inv_loss.item(), t_env)

        return repr_loss
    
    def compute_rew_loss(self, pred_rew, env_rew, mask):
        # pred_rew.shape: [bs, seq_len, 1]
        # mask.shape: [bs, seq_len, 1]
        mask = mask.squeeze(-1)
        rew_loss = F.mse_loss(pred_rew, env_rew, reduction="none").sum(-1)
        masked_rew_loss = (rew_loss * mask).sum() / mask.sum()
        return masked_rew_loss

    def compute_spr_loss(self, pred_f, target_f, mask):
        # pred_f.shape: [bs, seq_len, spr_dim]
        # mask.shape: [bs, seq_len, 1]
        mask = mask.squeeze(-1)
        spr_loss = F.mse_loss(pred_f, target_f, reduction="none").sum(-1)
        mask_spr_loss = (spr_loss * mask).sum() / mask.sum()
        return mask_spr_loss
    
    def compute_inv_loss(self, pred_a, target_a, mask):
        # pred_a.shape: [bs, seq_len, n_agents*n_actions]
        # mask.shape: [bs, seq_len, 1]
        mask = mask.squeeze(-1)
        act_loss = F.mse_loss(pred_a, target_a, reduction="none").sum(-1)
        mask_action_loss = (act_loss * mask).sum() / mask.sum()
        return mask_action_loss

    def rl_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, repr_loss):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            state_repr_t = self.mac.mask_enc_forward(batch, t=t)
            if not self.args.rl_signal:
                state_repr_t = state_repr_t.detach()
            agent_outs = self.mac.rl_forward(batch, state_repr_t, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            state_repr_t = self.target_mac.mask_enc_forward(batch, t=t)
            target_agent_outs = self.target_mac.rl_forward(batch, state_repr_t, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        if self.args.standardise_returns:
            target_max_qvals = target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        rl_loss = (masked_td_error ** 2).sum() / mask.sum() 
        # Compute tot loss
        tot_loss = rl_loss + self.args.repr_coef * repr_loss

        # Optimise
        self.optimiser.zero_grad()
        tot_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.mac.agent.momentum_update()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)
            self.mac.agent.momentum_update()

        if t_env % self.args.learner_log_interval == 0:
            self.logger.log_stat("rl_loss", rl_loss.item(), t_env)
            self.logger.log_stat("tot_loss", tot_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env) 
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Representation learning training
        time0 = time.time()
        repr_loss = self.repr_train(batch, t_env, episode_num)
        # RL training
        self.rl_train(batch, t_env, episode_num, repr_loss)
        time1 = time.time()
        print("time1",time1-time0)#0.25~0.3

    def test_encoder(self, batch: EpisodeBatch):
        # states.shape: [batch_size, seq_len, state_dim]
        states = batch["state"]
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # go through vae
        recons, z = [], []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            recons_t, _, z_t = self.mac.vae_forward(batch, t)
            recons.append(recons_t)
            z.append(z_t)
        # recons.shape: [batch_size, seq_len, state_repre_dim]
        recons = th.stack(recons, dim=1)
        z = th.stack(z, dim=1)

        encoder_result = {
            "recons": recons,
            "z": z,
            "states": states,
            "mask": mask,
        }
        th.save(encoder_result, os.path.join(self.args.encoder_result_direc, "result.pth"))

    def _update_targets_hard(self):
        # not quite good, but don't have bad effect
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        # not quite good, but don't have bad effect
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.latent_model.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.latent_model.state_dict(), "{}/latent_model.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.latent_model.load_state_dict(th.load("{}/latent_model.th".format(path), map_location=lambda storage, loc: storage))