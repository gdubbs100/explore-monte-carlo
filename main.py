import torch
import torch.distributions as dist

from samplers.base_sampler import SamplerResults
from samplers.metropolis_sampler import MetropolisSampler

if __name__ == "__main__":
    
    def make_gaussian_mean_model(samples, prior_dist, like_cov = None):

        like_cov = torch.eye(samples.shape[-1]) if like_cov is None else like_cov

        def log_prob(params):
            prior_lp = prior_dist.log_prob(params)
            like = dist.MultivariateNormal(params, like_cov)
            like_lp = like.log_prob(samples.unsqueeze(1)).sum(dim=0)
            return prior_lp + like_lp
        
        return log_prob

    ## Generate Synthetic Data
    n_samples = 5000
    base_dist = dist.MultivariateNormal(
        loc=torch.Tensor([1, 4]), 
        covariance_matrix=torch.eye(2)
    )

    samples = base_dist.sample((n_samples,))

    prior = dist.MultivariateNormal(torch.tensor([0.0, 0.0]), 10.0 * torch.eye(2))
    model = make_gaussian_mean_model(samples, prior)
    sampler = MetropolisSampler(
        model, 
        dim=2, 
        proposal_std=0.03, 
        init_dist=prior, 
        seed=0
    )

    num_iters, num_chains, burnin = 1000, 10, 0
    sampler.run(num_iters=num_iters, num_chains=num_chains, burnin=burnin)
    results = sampler.results
    print(isinstance(results, SamplerResults))
    breakpoint()
    print(torch.Tensor(results.params['posterior']['theta'].values).mean(dim=(0, 1)))
