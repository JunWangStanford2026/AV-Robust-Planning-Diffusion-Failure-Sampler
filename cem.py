import multiprocessing
import os
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import torch
import numpy as np
from torch.distributions import MultivariateNormal
from pomegranate.gmm import GeneralMixtureModel
from pomegranate import distributions as pgd
from tqdm import tqdm

def cross_entropy_method(f, d, px, m, m_elite, kmax):
    """
    Cross-entropy method for maximizing a function f: R^d -> R.
    Args:
        f: function to maximize
        d: distribution to sample from
        px: distribution to compute the importance weights
        m: number of samples per iteration
        m_elite: number of elite samples per iteration
        kmax: number of iterations
    Returns:
        d: final distribution
        x_best: best solution found
        f_best: best objective value found
    """

    # Save elite samples and their scores
    all_elite_samples = torch.zeros((kmax, m_elite, d.event_shape[0]))
    all_elite_scores = torch.zeros((kmax, m_elite))

    for k in range(kmax):
        samples = d.sample((m,))
        scores = torch.tensor([f(s) for s in samples])#f(samples)
        elite_samples = samples[scores.topk(min(m, m_elite)).indices]
        weights = torch.exp(px.log_prob(elite_samples.squeeze()) -  d.log_prob(elite_samples.squeeze()))
        weights = weights / weights.sum()
        #print(weights.min(), weights.max())
        mean = (elite_samples * weights.unsqueeze(1)).sum(dim=0)
        cov = torch.cov(elite_samples.T, aweights=weights)
        cov = cov + 1e-1*torch.eye(d.event_shape[0])
        #print(mean)
        #print(cov)
        d = MultivariateNormal(mean, cov)

        all_elite_samples[k] = elite_samples
        all_elite_scores[k] = scores.topk(min(m, m_elite)).values


    x_best = mean
    f_best = f(x_best)

    return mean, cov, x_best, f_best

# JITTER = 1e-1

# def compute_weights(px, model, elite_samples):
#     """Compute weights using logsumexp trick."""
#     log_weights = px.log_prob(elite_samples.squeeze()) -  model.log_probability(elite_samples.squeeze())
#     #max_log_weights = log_weights.max()
#     #weights = torch.exp(log_weights - max_log_weights)
#     #weights = weights / weights.sum()
#     #log_norm = torch.logsumexp(log_weights, dim=0)
#     #weights = torch.exp(log_weights - log_norm)
#     weights = torch.exp(log_weights)
    
#     return weights.clamp(min=1e-3)
#     #return weights

# def update_model_means(model, elite_samples, ncomponents):
#     """Update model means if any component has a nan mean."""
#     for i in range(ncomponents):
#         if torch.isnan(model.distributions[i].means).any():
#             model.distributions[i].means = torch.nn.Parameter(elite_samples.mean(dim=0))

# def add_jitter_to_covs(model, ncomponents):
#     """Add jitter to covariances to ensure they are positive definite."""
#     # if any component has a nan covariance, set it to identity
#     for i in range(ncomponents):
#         if model.distributions[i].covariance_type == "diag":
#             if torch.isnan(model.distributions[i].covs).any():
#                 model.distributions[i].covs =  torch.nn.Parameter(torch.ones(model.distributions[i].covs.shape[0]))
#         elif model.distributions[i].covariance_type == "full":
#             if torch.isnan(model.distributions[i].covs).any():
#                 model.distributions[i].covs =  torch.nn.Parameter(torch.eye(model.distributions[i].covs.shape[0]))


#     for i in range(ncomponents):
#         if model.distributions[i].covariance_type == "diag":
#             model.distributions[i].covs += torch.ones(model.distributions[i].covs.shape[0]) * JITTER
#             model.distributions[i].covs.data = model.distributions[i].covs.data.clamp(min=JITTER, max=1e2)
#         elif model.distributions[i].covariance_type == "full":
#             model.distributions[i].covs += torch.eye(model.distributions[i].covs.shape[0]) * JITTER
#             model.distributions[i].covs.data = model.distributions[i].covs.data.clamp(min=JITTER, max=1e2)

# def gmm_cross_entropy_method(sim_fn, rho_target, components, px, m, m_elite, kmax, save_dir, verbose=False):
#     """
#     Cross-entropy method for maximizing a function f: R^d -> R.
#     Args:
#         sim_fn: function to maximize
#         components: list of distributions to sample from
#         px: distribution to compute the importance weights
#         m: number of samples per iteration
#         m_elite: number of elite samples per iteration
#         kmax: number of iterations
#         save_dir: directory to save the samples, scores, and observations
#     """
#     ncomponents = len(components)
#     model = GeneralMixtureModel(components, verbose=verbose)
#     k = 0
#     rho = -np.Inf

#     while k < kmax:
#         samples = model.sample(m)

#         # scatter the samples in a pyplt
#         plt.scatter(samples[:, 0], samples[:, 1])
#         plt.savefig(f"scatter_{k}.png")

#         evaluations = [sim_fn(sample) for sample in samples]
#         scores = torch.tensor([e[0] for e in evaluations])
#         print(scores)
#         observations = [e[1] for e in evaluations]

#         elite_indices = scores.topk(min(m, m_elite)).indices
#         elite_samples = samples[elite_indices]
#         elite_scores = scores.topk(min(m, m_elite)).values
#         rho = np.min([elite_scores.min(), rho_target])
#         print(rho)

#         weights = compute_weights(px, model, elite_samples)
#         model = GeneralMixtureModel(model.distributions, verbose=verbose)
#         model = model.fit(elite_samples, sample_weight=weights)

#         update_model_means(model, elite_samples, ncomponents)
#         add_jitter_to_covs(model, ncomponents)

#         if verbose:
#             for dist in model.distributions:
#                 print(dist.means)
#                 print(dist.covs)

#         # Save samples, scores, and observations to the specified directory
#         torch.save(samples, os.path.join(save_dir, f'samples_{k}.pt'))
#         torch.save(scores, os.path.join(save_dir, f'scores_{k}.pt'))
#         torch.save(observations, os.path.join(save_dir, f'observations_{k}.pt'))

#         k += 1

#     return model








def gmm_cross_entropy_method(sim_fn, rho_target, components, px, m, m_elite, kmax, save_dir, verbose=False):
    """
    Cross-entropy method for maximizing a function f: R^d -> R.
    Args:
        f: function to maximize
        components: list of distributions to sample from
        px: distribution to compute the importance weights
        m: number of samples per iteration
        m_elite: number of elite samples per iteration
        kmax: number of iterations
    """
    # Save elite samples and their scores
    ncomponents = len(components)
    all_samples = []
    all_scores = []
    all_observations = []
    
    model = GeneralMixtureModel(components)
    rho = 0.0
    k = 0

    while k < kmax:# and rho < rho_target:
        samples = model.sample(m,)

        # evaluations = [sim_fn(sample) for sample in tqdm(samples)]
        evaluations = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(sim_fn)(x=samples[i]) for i in tqdm(range(samples.shape[0])))
        scores = torch.tensor([e[0] for e in evaluations])
        observations = [e[1] for e in evaluations]

        # processed_list = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(sim_fn)(x=samples[i]) for i in range(samples.shape[0]))
        # scores = torch.tensor([e[0] for e in processed_list])
        # observations = [e[1] for e in processed_list]

        elite_samples = samples[scores.topk(min(m, m_elite)).indices]
        elite_scores = scores.topk(min(m, m_elite)).values
        rho = np.min([elite_scores.min(), rho_target])
        print("rho: ", rho)
        log_weights = px.log_prob(elite_samples.squeeze()) -  model.log_probability(elite_samples.squeeze())
        #print(log_weights.shape)
        #weights = torch.exp(px.log_prob(elite_samples.squeeze()) -  model.log_probability(elite_samples.squeeze()))
        #weights = weights / weights.sum()

        # Compute weights with logsumexp trick
        max_log_weights = log_weights.max()
        weights = torch.exp(log_weights - max_log_weights)
        #weights.clamp(min=1e-1)
        weights = weights / weights.sum()
        #log_norm = torch.logsumexp(log_weights, dim=0)
        #weights = torch.exp(log_weights - log_norm)
        #weights = weights.clamp(min=1e-3)
        #weights = torch.exp(log_weights)

        # print weight stats
        if verbose:
            print("Weights:")
            print(weights.min(), weights.max())
            print(weights.mean(), weights.std())

        try:
            model = model.fit(elite_samples, sample_weight=weights)
        except:
            print("Error in fitting model, retry with new samples")
            continue
        #model = model.fit(elite_samples)
        
        if verbose:
            print("Priors:")
            print(model.priors)
            print("Means:")
            print(model.distributions[0].means)
            print(model.distributions[1].means)
            
            print("Covs:")
            print(model.distributions[0].covs)



        # compute next means and covs
        means = []
        covs = []
        for i in range(ncomponents):
            mean_i = model.distributions[i].means
            means.append(mean_i)

            # if covariance is nan, set it to the identity matrix
            # clamp covariance to ensure it is positive definite
            if model.distributions[i].covariance_type == "diag":
                if torch.isnan(model.distributions[i].covs).any():
                    cov_i = torch.ones(model.distributions[i].covs.shape[0])
                else:
                    cov_i = model.distributions[i].covs
                cov_i += torch.ones(model.distributions[i].covs.shape[0]) * 1e-1
                cov_i = cov_i.clamp(min=1e-1, max=1e1)
            elif model.distributions[i].covariance_type == "full":
                if torch.isnan(model.distributions[i].covs).any():
                    cov_i = torch.eye(model.distributions[i].covs.shape[0])
                else:
                    cov_i = model.distributions[i].covs
                cov_i += torch.eye(model.distributions[i].covs.shape[0]) * 1e-1
                cov_i = cov_i.clamp(min=1e-1, max=1e1)

            covs.append(cov_i)

        # update model means and covs
        priors = torch.tensor(model.priors.data)
        priors = torch.clamp(priors, min=1e-1)
        priors = priors / priors.sum()
        model = GeneralMixtureModel([pgd.Normal(means[i], covs[i], covariance_type=model.distributions[i].covariance_type) for i in range(ncomponents)], priors=priors)
            

                
        # Save samples, scores, and observations to the specified directory
        torch.save(samples, os.path.join(save_dir, f'samples-{k}.pt'))
        torch.save(scores, os.path.join(save_dir, f'risks-{k}.pt'))
        torch.save(observations, os.path.join(save_dir, f'observations-{k}.pt'))
        k += 1
        print("Iteration: ", k)

    return model