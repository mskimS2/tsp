import torch
import torch.nn as nn
from torch.distributions import Categorical

from models.layers import Glimpse, LinearEmbedding, Pointer


class AttentionBlock(nn.Module):
    def __init__(
        self, 
        embed_dim: int, 
        n_heads: int, 
        feed_forward_hidden: int = 512, 
        bn: bool = False
    ):
        super(AttentionBlock, self).__init__()
        self.mha = torch.nn.MultiheadAttention(embed_dim, n_heads)
        self.embed = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_hidden),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden, embed_dim)
        )

    def forward(self, x):
        # Multiheadattention in pytorch starts with (target_seq_length, batch_size, embedding_size).
        # thus we permute order first. https://pytorch.org/docs/stable/nn.html#multiheadattention
        x = x.permute(1, 0, 2)
        x = x + self.mha(x, x, x)[0]
        x = x.permute(1, 0, 2)
        x = x + self.embed(x)
        
        return x


class AttentionModule(nn.Sequential):
    def __init__(
        self, 
        embed_dim: int, 
        n_heads: int, 
        feed_forward_hidden: int = 512, 
        n_self_attentions:int = 2, 
        bn: bool = False
    ):
        super(AttentionModule, self).__init__(
            *(AttentionBlock(embed_dim, n_heads, feed_forward_hidden, bn) for _ in range(n_self_attentions))
        )


class AttentionTSP(nn.Module):
    def __init__(
        self,
        pos_size: int,
        embed_size: int,
        hidden_size: int,
        seq_len: int,
        n_head: int = 4,
        C: float = 10
    ):
        super(AttentionTSP, self).__init__()

        self.embedding_size = embed_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.n_head = n_head
        self.C = C

        self.embedding = LinearEmbedding(pos_size, embed_size)
        self.mha = AttentionModule(embed_size, n_head)

        self.init_w = nn.Parameter(torch.Tensor(pos_size * self.embedding_size))
        self.init_w.data.uniform_(-1, 1)
        self.glimpse = Glimpse(self.embedding_size,
                               self.hidden_size, self.n_head)
        self.pointer = Pointer(self.embedding_size,
                               self.hidden_size, 1, self.C)

        self.h_context_embed = nn.Linear(
            self.embedding_size, self.embedding_size)
        self.v_weight_embed = nn.Linear(
            self.embedding_size * pos_size, self.embedding_size)

    def forward(self, inputs):
        # inputs: [batch_size x seq_len x 2]

        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        embedded = self.embedding(inputs)
        h = self.mha(embedded)
        h_mean = h.mean(dim=1)
        h_bar = self.h_context_embed(h_mean)
        h_rest = self.v_weight_embed(self.init_w)
        query = h_bar + h_rest

        # init query
        prev_chosen_indices = []
        prev_chosen_logprobs = []
        first_chosen_hs = None
        mask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool)

        for index in range(self.seq_len):
            _, n_query = self.glimpse(query, h, mask)
            prob, _ = self.pointer(n_query, h, mask)
            cat = Categorical(prob)
            chosen = cat.sample()
            logprobs = cat.log_prob(chosen)
            prev_chosen_indices.append(chosen)
            prev_chosen_logprobs.append(logprobs)

            mask[[i for i in range(batch_size)], chosen] = True
            cc = chosen.unsqueeze(1).unsqueeze(2).repeat(
                1, 1, self.embedding_size
            )
            if first_chosen_hs is None:
                first_chosen_hs = h.gather(1, cc).squeeze(1)
            chosen_hs = h.gather(1, cc).squeeze(1)
            h_rest = self.v_weight_embed(
                torch.cat([first_chosen_hs, chosen_hs], dim=-1))
            query = h_bar + h_rest

        return torch.stack(prev_chosen_logprobs, 1), torch.stack(prev_chosen_indices, 1)
