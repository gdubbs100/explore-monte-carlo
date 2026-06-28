import torch
import torch.distributions as dist

from samplers.metropolis_sampler import MetropolisSampler
from experiment_setup.gaussian_experiment import (
    sample_gaussian_mean_data,
    make_gaussian_mean_model
)

from result_logging.utils import (
    extract_chain_statistics_from_results,
    extract_parameter_statistics_from_results,
    plot_distribution_by_sampler,
    plot_param_convergence
)

if __name__ == "__main__":
    n_samples = 500
    n_dims = 5

    samples, locations = sample_gaussian_mean_data(
        n_samples = n_samples,
        n_dims = n_dims
    )

    prior = dist.MultivariateNormal(torch.zeros_like(locations), torch.eye(n_dims))
    model = make_gaussian_mean_model(samples, prior)

    num_iters, num_chains, burnin = 1000, 10, 0

    sampler1 = MetropolisSampler(
        model, 
        dim=n_dims, 
        proposal_std=0.03, 
        init_dist=prior, 
        seed=0
    )

    sampler1.run(num_iters=num_iters, num_chains=num_chains, burnin=burnin)

    sampler2 = MetropolisSampler(
        model, 
        dim=n_dims, 
        proposal_std=0.2, 
        init_dist=prior, 
        seed=0
    )

    sampler2.run(num_iters=num_iters, num_chains=num_chains, burnin=burnin)

    all_sampler_results = {"sampler1": sampler1.results, "sampler2": sampler2.results}
    param_sampler_results_df = extract_parameter_statistics_from_results(all_sampler_results=all_sampler_results)
    chain_sampler_results_df = extract_chain_statistics_from_results(all_sampler_results=all_sampler_results)

    fig1 = plot_distribution_by_sampler(
        sampler_results_df=param_sampler_results_df,
        y_axis = "bulk_ess_per_second",
        title = "Bulk ESS per second - distribution over parameters",
        xlabel = "Samplers",
        ylabel = "Bulk ESS per second"
    )
    fig1.savefig('./logs/fig1.png')

    fig2 = plot_distribution_by_sampler(
        sampler_results_df=param_sampler_results_df,
        y_axis = "tail_ess_per_second",
        title = "Tail ESS per second - distribution over parameters",
        xlabel = "Samplers",
        ylabel = "Tail ESS per second"
    )
    fig2.savefig('./logs/fig2.png')

    fig3 = plot_distribution_by_sampler(
        sampler_results_df=param_sampler_results_df,
        y_axis = "rhat",
        title = "Rhat - distribution over parameters",
        xlabel = "Samplers",
        ylabel = "Rhat"
    )
    fig3.savefig('./logs/fig3.png')

    fig4 = plot_distribution_by_sampler(
        sampler_results_df=chain_sampler_results_df,
        y_axis = "acceptance_rate",
        title = "Acceptance Rate by chain",
        xlabel = "Samplers",
        ylabel = "Acceptance Rate"
    )
    fig4.savefig('./logs/fig4.png')

    fig5 = plot_param_convergence(
        params = sampler1.results.params, 
        true_params = locations,
        num_chains = num_chains
    )
    fig5.savefig('./logs/fig5.png')

    fig6 = plot_param_convergence(
        params = sampler2.results.params, 
        true_params = locations,
        num_chains = num_chains
    )
    fig6.savefig('./logs/fig6.png')

