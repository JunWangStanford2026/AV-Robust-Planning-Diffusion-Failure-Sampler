from typing import Callable, Optional
from pathlib import Path
from multiprocessing import cpu_count
import multiprocessing
import torch
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from accelerate import Accelerator, DataLoaderConfiguration
from ema_pytorch import EMA

from difs.diffusion import GaussianDiffusionConditional
from difs.dataset import DatasetConditional
from difs.utils import exists, cycle

import wandb
from tqdm.auto import tqdm
import concurrent.futures
from joblib import Parallel, delayed
import os

N_CORES = multiprocessing.cpu_count()

class DiFS(object):
    def __init__(
        self,
        diffusion_model,
        evaluate_fn: Callable, # should take in a tensor of shape (B, C, T) where B is batch-size, C is number of channels, and T is lenght of time series. Return a tuple of two tensors: the first tensor should be the robustness value, and the second tensor should be the state trajectory.
        init_disturbances: torch.Tensor,
        envs_list: list,
        run_serial: int,
        *,
        alpha: float = 0.5,
        N: int,
        max_iters: int = 200,
        train_num_steps: int = 10000,
        train_batch_size: int = 256,
        sample_batch_size: int = 10000,
        gradient_accumulate_every: int = 1,
        train_lr: float = 1e-3,
        ema_update_every: float = 10,
        ema_decay: float = 0.995,
        amp: bool = False,
        mixed_precision_type: str = 'fp16',
        split_batches: bool = True,
        max_grad_norm: float = 1.,
        experiment_name: str = "test",
        results_folder: str = './results',
        sample_callback: Optional[Callable] = None,
        use_wandb: bool = False,
        wandb_plot_fn: Optional[Callable] = None,
        is_training: bool = True,
        save_intermediate: bool = False,
    ):
        super().__init__()
        self.init_disturbances = init_disturbances  # shape (N, 4, 24)
        self.envs_list = envs_list
        self.rho_target = 0.0
        self.alpha = alpha
        self.N = N
        self.evaluate_fn = evaluate_fn
        self.train_batch_size = train_batch_size
        self.cond_dim = diffusion_model.model.cond_dim
        self.sample_batch_size = sample_batch_size
        self.use_wandb = use_wandb
        self.wandb_plot_fn = wandb_plot_fn
        self.max_iters = max_iters
        self.save_intermediate = save_intermediate
        self.run_serial = run_serial

        # accelerator
        self.accelerator = Accelerator(
            dataloader_config = DataLoaderConfiguration(split_batches=split_batches),
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model
        self.model = diffusion_model
        self.channels = diffusion_model.channels

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        # optimizer
        # self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.opt = AdamW(diffusion_model.parameters(), lr=train_lr, weight_decay=1e-5)

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)
        
        # for logging results in a folder periodically
        # if is_training:
        #     self.results_folder = Path(results_folder+"/"+experiment_name)
        #     self.results_folder.mkdir(parents=True, exist_ok = True)

        # step counter state
        self.step = 0

        # prepare model, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        self.sample_callback = sample_callback

    @property
    def device(self):
        return self.accelerator.device

    def save(self, k):
        if not self.accelerator.is_local_main_process:
            return

        save_data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        print(os.path.expanduser(k))

        torch.save(save_data, os.path.expanduser(k))

    def load(self, k):
        accelerator = self.accelerator
        device = accelerator.device

        if type(k) == str:
            data = torch.load(k, map_location=device)
        else:
            data = torch.load(str(self.results_folder / f'model-{k}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def log_wandb(self, rho, data, conds, inits, samples, robustness, elite_samples, elite_rs, save_path):
        info = {
            "rho": rho,
            "dataset_size": data.shape[0],
            "initial_dx_std": inits[:, 0].std().cpu(),
            "initial_dy_std": inits[:, 1].std().cpu(),
            "initial_dvx_std": inits[:, 2].std().cpu(),
            "initial_dvy_std": inits[:, 3].std().cpu(),
            "mean_sampled_robustness": robustness.mean().cpu(),
            "min_sampled_robustness": robustness.min().cpu(),
            "max_sampled_robustness": robustness.max().cpu(),
            "std_sampled_robustness": robustness.std().cpu(),
            "mean_elite_robustness": elite_rs.mean().cpu(),
            "min_elite_robustness": elite_rs.min().cpu(),
            "max_elite_robustness": elite_rs.max().cpu(),
            "std_elite_robustness": elite_rs.std().cpu(),
        }

        if self.wandb_plot_fn is not None:
            fig = self.wandb_plot_fn(observations, robustness, self.rho_target)
            info["samples"] = wandb.Image(fig)
            
        wandb.log(info)

        if info["mean_sampled_robustness"] <= 5.0 and rho <= 1.0:
            self.save(save_path)


    def sample(self, rho_sample, inits, no_grad=False):
        # inits must have shape (num samples to draw, 4)
        rho_sample = rho_sample.unsqueeze(1).repeat(1, self.model.cond_dim).to(self.device)
        if rho_sample.shape[0] > self.sample_batch_size:
            rho_sample_tuple = torch.split(rho_sample, self.sample_batch_size, dim=0)
            inits_tuple = torch.split(inits, self.sample_batch_size, dim=0)
            # samples = [self.model.sample(rs, no_grad) for rs in rho_sample_tuple]
            samples = [self.model.sample(rho_sample_tuple[i], no_grad, inits_tuple[i]) for i in range(len(rho_sample_tuple))]
            samples = torch.cat(samples, dim=0)
        else:
            samples = self.model.sample(rho_sample, no_grad, inits)
        return samples

    def training_loop(self, data, conds, inits, update_steps=None):
        accelerator = self.accelerator
        device = accelerator.device
        
        dataset = DatasetConditional(data.cpu(), conds.cpu(), inits.cpu())
        dl = DataLoader(dataset, batch_size = self.train_batch_size, pin_memory = True, num_workers = 0)
        dl = self.accelerator.prepare(dl)
        dl_cycle = cycle(dl)

        print("Training on updated data with {} samples...".format(data.shape[0]))
        step=0
        if update_steps == None:
            update_steps = self.train_num_steps
        with tqdm(initial = step, total = update_steps, disable = not accelerator.is_main_process) as pbar:
                while step < self.train_num_steps:

                    total_loss = 0.

                    for _ in range(self.gradient_accumulate_every):
                        databatch, condbatch, initbatch = next(dl_cycle)
                        databatch.to(device)
                        condbatch.to(device)
                        initbatch.to(device)

                        with self.accelerator.autocast():
                            loss = self.model(databatch, condbatch, initbatch)
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()

                        self.accelerator.backward(loss.to(torch.float32))

                    pbar.set_description(f'loss: {total_loss:.4f}')
                    wandb.log({'loss': total_loss})

                    accelerator.wait_for_everyone()
                    accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)


                    self.opt.step()
                    self.opt.zero_grad()

                    accelerator.wait_for_everyone()

                    step += 1
                    if accelerator.is_main_process:
                        self.ema.update()

                    pbar.update(1)

    def pretrain(self, data, cond, inits):
        self.training_loop(data, cond, inits, update_steps=50000)

    def train(self, ask_every=100):
        accelerator = self.accelerator
        device = accelerator.device

        # Initial observation disturbance samples
        print("Loading initial disturbance samples...")
        data = self.init_disturbances.clone().detach()
        sample_trajectories = [data[i] for i in range(data.shape[0])]

        # Get the starting states of the environments
        print("Initializing environments & Getting initial states...")
        inits = torch.empty((len(sample_trajectories), 4))
        for i in range(len(sample_trajectories)):
            env = self.envs_list[i]
            road = env.unwrapped.road
            intruder_vehicle = road.vehicles[0]
            ego_vehicle = road.vehicles[1]
            inits[i, :2] = torch.from_numpy(intruder_vehicle.position - ego_vehicle.position).to(inits.device)
            inits[i, 2:] = torch.from_numpy(intruder_vehicle.velocity - ego_vehicle.velocity).to(inits.device)

        print("Evaluating robustness of initial disturbance samples...")
        simulations = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.evaluate_fn, sample_trajectories[i].clone().detach().numpy(), self.envs_list[i]) for i in range(len(sample_trajectories))]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                simulations.append(future.result())

        conds = torch.tensor(simulations).unsqueeze(1).repeat(1, 24)

        print("Training on initial data...")
        self.training_loop(data, conds, inits)

        rho = torch.quantile(conds, 1-self.alpha).item()

        k = 0
        while rho > self.rho_target and k <= self.max_iters:
            # Sample from the model with conditions [0, rho]
            print("rho = {}".format(rho))
            print("resample robustness conditions...")
            rho_sample = torch.distributions.Uniform(-0.1, rho).sample((self.N,))
            rho_sample = torch.clamp(rho_sample, min=0.0)

            # Reinitialize the simulation environments, and save the new initial states
            print("reinitializing simulation environments...")
            new_inits = torch.empty((self.N, 4), device=device)
            for i in range(self.N):
                env = self.envs_list[i]
                env.reset()
                road = env.unwrapped.road
                intruder_vehicle = road.vehicles[0]
                ego_vehicle = road.vehicles[1]
                new_inits[i, :2] = torch.from_numpy(intruder_vehicle.position - ego_vehicle.position).to(new_inits.device)
                new_inits[i, 2:] = torch.from_numpy(intruder_vehicle.velocity - ego_vehicle.velocity).to(new_inits.device)

            # Draw samples
            print("Drawing samples with diffusion model...")
            samples = self.sample(rho_sample, new_inits).clone().detach().cpu()
            print("DISTURBANCE SCALE OF CURRENT BATCH: ", torch.mean(torch.abs(samples).to(torch.float)))
            print("DISTURBANCE MEAN: ", torch.mean(samples.to(torch.float)))
            print("DISTURBANCE STD: ", torch.std(samples.to(torch.float)))

            # Evaluate robustness of each sample
            print("Evaluating robustness of samples drawn...")
            sample_trajectories = [samples[i] for i in range(samples.shape[0])]
            simulations = []
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(self.evaluate_fn, sample_trajectories[i].clone().detach().numpy(), self.envs_list[i]) for i in range(len(sample_trajectories))]
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    simulations.append(future.result())

            robustness = torch.tensor(simulations).unsqueeze(1).repeat(1, 24)
            if self.save_intermediate:
                torch.save(robustness[:, 0], "./logs/sample_robustnesses_iter_" + str(k) + ".pt")


            # Compute the (1-alpha) quantile of robustness
            rho = torch.quantile(robustness, 1-self.alpha).item()
            rho = max(rho, self.rho_target)
            print("rho = {}".format(rho))

            if (k != 0 and k % ask_every == 0):
                if input("Save this pretrained model?(Y/N):") == "Y":
                    # self.save("~/Desktop/REU/Nissan/Code/Distillation_Models/distillation_1_steps_difs_pretrained_iter_" + str(k) + ".pt")
                    self.save(input("Enter file path: "))
                if input("Continue training (Y/N)?:") == "N":
                    return 

            # Add samples and conditions to dataset
            data = torch.cat((data, samples.cpu()), dim=0)
            conds = torch.cat((conds, robustness.cpu()), dim=0)
            inits = torch.cat((inits, new_inits.cpu()), dim=0)

            print("Selecting elite samples for training")
            mask = conds[:, 0] <= rho

            save_path = "./models/run_" + str(self.run_serial) + "_iter_" + str(k) + ".pt"
            # Logging to wandb
            if self.use_wandb and wandb.run is not None:
                self.log_wandb(rho, data, conds, inits, samples, robustness, data[mask, :, :], conds[mask, :], save_path)

            if self.save_intermediate:
                torch.save(conds[mask, 0], "./logs/elite_robustnesses_iter_" + str(k) + ".pt")

            # train
            self.training_loop(data[mask, :, :], conds[mask, :], inits[mask, :])

            k += 1