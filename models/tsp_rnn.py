import math
import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical

from models.layers import Attention, LinearEmbedding


class RNNTSP(nn.Module):
    def __init__(
        self,
        pos_size: int,
        embed_size: int,
        hidden_size: int,
        seq_len: int,
        n_glimpses: int,
        tanh_exploration: float
    ):
        super(RNNTSP, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_glimpses = n_glimpses
        self.seq_len = seq_len

        self.embedding = LinearEmbedding(pos_size, embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.pointer = Attention(hidden_size, C=tanh_exploration)
        self.glimpse = Attention(hidden_size)

        self.dec_sos = nn.Parameter(torch.FloatTensor(embed_size))
        self.dec_sos.data.uniform_(-(1. / math.sqrt(embed_size)),
                                   1. / math.sqrt(embed_size))

    def apply_mask_to_logits(self, logits, mask, idxs):
        batch_size = logits.size(0)
        clone_mask = mask.clone()
        if idxs is not None:
            clone_mask[[i for i in range(batch_size)], idxs.data] = 1
            logits[clone_mask] = -np.inf

        return logits, clone_mask

    def forward(self, inputs):
        # inputs: (batch_size x seq_len x 2)
        
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        embedded = self.embedding(inputs)
        encoder_outputs, (hidden, context) = self.encoder(embedded)

        prev_chosen_logprobs = []
        preb_chosen_indices = []
        mask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool)

        decoder_input = self.dec_sos
        decoder_input = decoder_input.unsqueeze(0).repeat(batch_size, 1)
        for index in range(seq_len):
            _, (hidden, context) = self.decoder(
                decoder_input.unsqueeze(1), (hidden, context)
            )

            query = hidden.squeeze(0)
            for _ in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                _mask = mask.clone()
                # logits[_mask] = -100000.0
                logits[_mask] = -np.inf
                query = torch.matmul(
                    ref.transpose(-1, -2),
                    torch.softmax(logits, dim=-1).unsqueeze(-1)
                ).squeeze(-1)

            _, logits = self.pointer(query, encoder_outputs)

            _mask = mask.clone()
            logits[_mask] = -np.inf
            probs = torch.softmax(logits, dim=-1)
            cat = Categorical(probs)
            chosen = cat.sample()
            mask[[i for i in range(batch_size)], chosen] = True
            log_probs = cat.log_prob(chosen)
            decoder_input = embedded.gather(
                1, chosen[:, None, None].repeat(1, 1, self.hidden_size)
            ).squeeze(1)
            prev_chosen_logprobs.append(log_probs)
            preb_chosen_indices.append(chosen)

        return torch.stack(prev_chosen_logprobs, 1), torch.stack(preb_chosen_indices, 1)
