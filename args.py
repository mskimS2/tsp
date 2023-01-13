import torch
import argparse


def get_args():
    p = argparse.ArgumentParser('reinforcement learning based rnn tsp')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                   help='choice the deivce (cpu or cuda)')
    p.add_argument('--pos_size', type=int, default=2,
                   help='dimension(x, y)')
    p.add_argument('--seq_len', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num_epochs', type=int, default=100)
    p.add_argument('--num_train', type=int, default=30000)
    p.add_argument('--num_test', type=int, default=2000)
    p.add_argument('--embed_size', type=int, default=128)

    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--hidden_size', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--grad_clip', type=float, default=0.8)
    p.add_argument('--beta', type=float, default=0.9)
    p.add_argument('--random_seed', type=int, default=2022)

    return p.parse_args()
