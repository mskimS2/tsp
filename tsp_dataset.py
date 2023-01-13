import torch
from torch.utils.data import DataLoader, Dataset

from generate_tsp import get_tsp_solution, generator_tsp_coord
from draw_tsp import draw_tsp


class TSPDataset(Dataset):
    def __init__(self, num_nodes, num_samples):
        super(TSPDataset, self).__init__()

        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.pointset = generator_tsp_coord(
            num_nodes=num_nodes,
            num_samples=num_samples
        )

        self.tsp_solution = [get_tsp_solution(
            points) for points in self.pointset]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        points = self.pointset[index]
        solution = self.tsp_solution[index]

        return index, points


def get_trainloader(args, dataset):
    train_data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True
    )

    return train_data_loader


def get_testloader(args, dataset):
    eval_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )

    return eval_loader


if __name__ == "__main__":
    """
    test code
    """
    num_nodes = 30
    num_samples = 5
    train_loader = DataLoader(
        TSPDataset(num_nodes, num_samples),
        batch_size=64,
        shuffle=True,
        num_workers=1
    )

    for idx, (a, b) in enumerate(train_loader):
        print(f"batch_count : {idx+1}")
        print(f"pointset : {len(a)}")
        print(f"solution : {len(b)}")

    print("TSP test figure")
    for (points, solutions) in train_loader:
        for p, s in zip(points, solutions):
            print(f"pointset : {p}")
            print(f"solution : {s}")
            draw_tsp(p, s)
