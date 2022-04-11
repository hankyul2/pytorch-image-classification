from pic.utils import Metric
from torch.utils.data import DataLoader

def compute_mean_std(dataset):
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)

    mean_m = Metric(header='mean')
    std_m = Metric(header='std')
    for x, y in data_loader:
        mean_m.update(x.mean(dim=[0, -1, -2]))
        std_m.update(x.std(dim=[0, -1, -2]))

    mean = mean_m.compute()
    std = std_m.compute()

    print(f"mean: {mean} std: {std}")