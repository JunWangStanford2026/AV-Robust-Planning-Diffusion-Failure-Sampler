import torch
import numpy as np
from tqdm import tqdm
from prdc import compute_prdc

MC_OBS_PATH = "./samples/monte_carlo/failure_trajectories_EAST.pt"
OBS_PATH = "./samples/distillation/run_57_seed_2_EAST_distilled-obs.pt"
RISKS_PATH = "./samples/distillation/run_57_seed_2_EAST_distilled-risks.pt"

def density(x,x_true):
    return compute_prdc(real_features=x_true, fake_features=x, nearest_k=5)["density"]

def coverage(x,x_true):
    return compute_prdc(real_features=x_true, fake_features=x, nearest_k=5)["coverage"]
    

#MC CI 
try: 
    mc_trajectories = torch.stack(torch.load(MC_OBS_PATH)).reshape(-1,96*2)
except:
    mc_trajectories = torch.stack([torch.from_numpy(o) for o in torch.load(MC_OBS_PATH)]).reshape(-1,96*2)


#DiFS CI

try:
    difs_risks = torch.from_numpy(torch.load(RISKS_PATH))
except:
    difs_risks = torch.load(RISKS_PATH)
failure_idx = torch.where(difs_risks==0.0)[0]
# difs_trajectories = torch.Tensor(torch.load("./distillation_samples/C-model-14-distilled-1-steps-iter-71-1000-obs.pt"))[failure_idx].reshape(-1,96*2)
try:
    difs_trajectories = torch.stack([torch.from_numpy(o) for o in torch.load(OBS_PATH)])[failure_idx].reshape(-1,96*2)
except:
    difs_trajectories = torch.load(OBS_PATH)[failure_idx].reshape(-1,96*2)

no_mc = mc_trajectories.shape[0]
no_difs = difs_trajectories.shape[0]
mc_idx = np.arange(no_mc)
np.random.shuffle(mc_idx)
difs_idx = np.arange(no_difs)
np.random.shuffle(difs_idx)
metrics = compute_prdc(real_features=mc_trajectories[mc_idx[:min(no_difs, no_mc)]],fake_features=difs_trajectories[difs_idx[:min(no_difs, no_mc)]],nearest_k=5)
print("failure rate: ", failure_idx.shape[0]/difs_risks.shape[0])
print("density: ", metrics['density'])
print("coverage: ", metrics['coverage'])
