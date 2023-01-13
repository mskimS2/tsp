import torch
from torch import nn
from torch.autograd import Variable


class TSPSolver(nn.Module):
    def __init__(self, actor: nn.Module, device: str = 'cpu'):
        super(TSPSolver, self).__init__()
        self.actor = actor
        self.device = device

    def compute_reward(self, sample_solution):
        # sample_solution seq_len of (batch_size)
        # torch.LongTensor (batch_size x seq_len x 2)

        batch_size, seq_len, _ = sample_solution.size()
        tour_length = Variable(torch.zeros([batch_size])).to(self.device)

        # tour (0 -> 1 -> ... -> n-1 -> n) reward
        for i in range(seq_len - 1):
            tour_length += torch.norm(
                sample_solution[:, i, :] - sample_solution[:, i + 1, :], dim=-1
            )
        # last tour (n -> 0) reward
        tour_length += torch.norm(
            sample_solution[:, seq_len - 1, :] - sample_solution[:, 0, :], dim=-1
        )

        return tour_length

    def forward(self, inputs: torch.tensor):
        # inputs shape: (batch_size, input_size, seq_len)
        probs, actions = self.actor(inputs)
        R = self.compute_reward(
            inputs.gather(1, actions.unsqueeze(2).repeat(1, 1, 2))
        )

        return R, probs, actions
