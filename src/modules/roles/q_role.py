import torch.nn as nn
import torch.nn.functional as F

import torch as th


class QRole(nn.Module):
    def __init__(self, args):
        super(QRole, self).__init__()
        self.args = args
        self.n_actions = args.n_actions

        self.q_fc = nn.Linear(args.rnn_hidden_dim, self.n_actions)
        self.action_space = th.ones(args.n_actions).to(args.device)

    def forward(self, h, action_latent):
        q = self.q_fc(h)
        return q

    def update_action_space(self, new_action_space):
        self.action_space = th.Tensor(new_action_space).to(self.args.device).float()
