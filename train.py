import torch
import numpy as np
from tqdm import tqdm

from utils import set_randomness
from models.solver import TSPSolver
from models.tsp_rnn import RNNTSP
from models.tsp_attention import AttentionTSP
from tsp_dataset import TSPDataset, get_trainloader, get_testloader
from args import get_args
from tsp_heuristic import get_ref_reward


if __name__ == "__main__":
    args = get_args()
    set_randomness(args.random_seed)

    actor = RNNTSP(
        pos_size=args.pos_size,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        seq_len=args.seq_len,
        n_glimpses=2,
        tanh_exploration=10
    )
    # actor = AttentionTSP(
    #     embed_dim=args.embed_size,
    #     hidden_size=args.hidden_size,
    #     seq_len=args.seq_len,
    # )

    train_dataset = TSPDataset(args.seq_len, args.num_train)
    test_dataset = TSPDataset(args.seq_len, args.num_test)
    train_data_loader = get_trainloader(args, train_dataset)
    eval_loader = get_testloader(args, test_dataset)

    model = TSPSolver(actor=actor, device=args.device).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # Calculating heuristics
    heuristic_distance = torch.zeros(args.num_test)
    for i, pointset in tqdm(test_dataset):
        heuristic_distance[i] = get_ref_reward(pointset)
    # np.save(heuristic_distance.detach().numpy(),
    #         'save/heuristic_distance.npy')

    # generating first baseline
    moving_avg = torch.zeros(args.num_train, device=args.device)
    for (indices, sample_batch) in tqdm(train_data_loader):
        sample_batch = sample_batch.to(args.device)
        rewards, _, _ = model(sample_batch)
        moving_avg[indices] = rewards

    # Training
    best_length = 1e9
    for epoch in range(args.num_epochs):
        model.train()
        for batch_idx, (indices, sample_batch) in enumerate(train_data_loader):
            sample_batch = sample_batch.to(args.device)
            rewards, log_probs, action = model(sample_batch)
            moving_avg[indices] = (
                moving_avg[indices] * args.beta +
                rewards * (1.0 - args.beta)
            )
            advantage = rewards - moving_avg[indices]
            log_probs = torch.sum(log_probs, dim=-1)
            loss = (advantage * log_probs).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        # validate
        model.eval()
        ret = []
        for i, batch in eval_loader:
            batch = batch.to(args.device)
            R, _, _ = model(batch)
            ret.extend(R.detach().cpu().numpy())
        ret = np.array(ret)

        R = (ret / heuristic_distance).mean().detach().numpy()
        print(
            f"[{epoch}/epoch] model/heuristics: {R}, average tour length: {np.mean(ret)}")
        print(
            f"[{epoch}/epoch] model/heuristics: {R}, heuristic tour length: {np.mean(heuristic_distance)}")

        scheduler.step()

        if best_length > R:
            best_length = R
            torch.save(model.state_dict(), f'save/rnn{args.seq_len}.pth')

    print('model best_length: ', R)
