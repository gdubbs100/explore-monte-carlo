import torch

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from samplers.base_sampler import SamplerResults

def extract_parameter_statistics_from_results(all_sampler_results = dict[str, SamplerResults]) -> pd.DataFrame:
    """
    Extract ESS per second and Rhat from sampler results objects for a dict of Sampler results.
    Key of dict should be string naming the sampler
    """

    df = pd.DataFrame(
        {
            "bulk_ess_per_second": {k: v.bulk_ess_per_second.numpy() for k, v in all_sampler_results.items()},
            "tail_ess_per_second": {k: v.tail_ess_per_second.numpy() for k, v in all_sampler_results.items()},
            "rhat": {k : v.rhat.numpy() for k, v in all_sampler_results.items()}
        }
    ).explode(["bulk_ess_per_second", "tail_ess_per_second", "rhat"]).reset_index().rename(columns = {'index': "sampler"})

    return df

def extract_chain_statistics_from_results(all_sampler_results = dict[str, SamplerResults]) -> pd.DataFrame:
    """
    Extract Acceptance Rate metrics from sampler results
    Key of dict should be string naming the sampler
    """
    df = pd.DataFrame(
        {
            "acceptance_rate": {k: v.acceptance_rate.numpy() for k, v in all_sampler_results.items()},
        }
    ).explode(["acceptance_rate"]).reset_index().rename(columns = {'index': "sampler"})

    return df

def plot_distribution_by_sampler(sampler_results_df: pd.DataFrame, y_axis: str, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize = (8, 5))
    sns.boxplot(
        data = sampler_results_df,
        x = "sampler",
        y = y_axis,
        hue = "sampler",
        ax = ax
    )
    ax.set(title = title, xlabel = xlabel, ylabel = ylabel)
    plt.tight_layout()
    return fig

def plot_param_convergence(params: torch.Tensor, true_params: torch.Tensor, num_chains: int):
    # params has n_chains x n_iters x n_dims
    # true_params has n_dims
    # should broadcast true_params to match params
    errors = torch.pow(params - true_params, 2).numpy()
    fig, ax = plt.subplots(figsize = (8, 5))
    for c in range(num_chains):
        to_plot = errors[c, :, :]
        plt.plot(to_plot, c = 'blue', alpha = 0.2)
    
    return fig
