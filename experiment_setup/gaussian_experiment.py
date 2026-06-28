import torch
import torch.distributions as dist

def sample_gaussian_mean_data(
        n_samples: int, 
        n_dims: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a Gaussian distribution with means to be learnt using MCMC simulation
    """
    locations = torch.randn(n_dims)
    base_dist = dist.MultivariateNormal(
        loc=locations, 
        covariance_matrix=torch.eye(n_dims)
    )

    samples = base_dist.sample((n_samples,))
    return samples, locations

def make_gaussian_mean_model(
        samples: torch.Tensor, 
        prior_dist: dist.Distribution, 
        like_cov: torch.Tensor | None = None
    ) -> torch.Tensor:

    like_cov = torch.eye(samples.shape[-1]) if like_cov is None else like_cov

    def log_prob(params):
        prior_lp = prior_dist.log_prob(params)
        like = dist.MultivariateNormal(params, like_cov)
        like_lp = like.log_prob(samples.unsqueeze(1)).sum(dim=0)
        return prior_lp + like_lp
    
    return log_prob

