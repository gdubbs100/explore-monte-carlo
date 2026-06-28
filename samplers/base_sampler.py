import time

import torch
import torch.distributions as dist

from arviz import from_dict, ess, rhat

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Callable

@dataclass
class SamplerState:
    """Per-chain state the kernel carries between steps."""
    params: torch.Tensor          # (num_chains, dim) -- the position we log
    log_prob: torch.Tensor        # (num_chains,) cached log prob
    extras: dict = field(default_factory=dict)  # momentum, step_size, etc.

@dataclass
class SamplerResults:
    """Structured output to return results from MCMCSampler runs"""
    params: torch.Tensor            # (num_chains, num_iters -(burnin + 1), dim) -- parameter values per iteration 
    acceptance_rate: torch.Tensor   # (num_chains, dim) -- acceptance rate per chain
    bulk_ess_per_second: torch.Tensor  # bulk effective sample size per second for each parameter
    tail_ess_per_second: torch.Tensor  # tail effective sample size per second for each parameter
    rhat: torch.Tensor        # rhat or gelman_rubin statistic per parameter
    wall_time: float                # wall time in seconds for time to run sampling algorithm
    burnin: int                     # number of burnin iters to help plotting
    diagnostics: dict[torch.Tensor] # additional algorithm specific metrics


class MCMCSampler(ABC):

    def __init__(
            self,
            log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
            dim: int,
            init_dist: dist.Distribution,
            seed: int = 42
    ):
        self.log_prob_fn = log_prob_fn
        self.dim = dim
        self.init_dist = init_dist
        torch.manual_seed(seed)
        # results filled in when sampler's run method is called
        self.results = None

    @abstractmethod
    def step(self, state: SamplerState) -> tuple[SamplerState, dict]:
        """Step all chains forward one iteration. Return (new_state, info_dict)."""
        ...

    def init_state(self, num_chains: int) -> SamplerState:
        params = self.init_dist.sample((num_chains,)).reshape(num_chains, self.dim)
        return SamplerState(params=params, log_prob=self.log_prob_fn(params))

    def run(self, num_iters: int, num_chains: int, burnin: int = 0) -> None:
        assert burnin < num_iters, f"burnin (got {burnin}) must be less than num_iters (got {num_iters})"

        start_time = time.perf_counter()
        state = self.init_state(num_chains)
        history = torch.empty((num_chains, num_iters, self.dim))
        diagnostics = defaultdict(list)

        for t in range(num_iters):
            state, info = self.step(state)
            history[:, t, :] = state.params
            
            ## variant specific results
            for k, v in info.items():
                diagnostics[k].append(v)

        end_time = time.perf_counter()
        wall_time = end_time - start_time

        ## create run summary
        self.results = self.calculate_summary_statistics(
            history = history,
            burnin=burnin,
            diagnostics=diagnostics,
            wall_time=wall_time
        )
    
    def calculate_summary_statistics(self, history: torch.Tensor, burnin: int, diagnostics: dict, wall_time: float):
        params = history[:, burnin:, :]
        inference_data = from_dict(posterior={"theta": history[:, burnin:, :].numpy()})
        final_bulk_ess = torch.Tensor(
            ess(inference_data, var_names = ["theta"], method = "bulk")["theta"].values
        )

        final_bulk_ess_per_second = final_bulk_ess / wall_time

        final_tail_ess = torch.Tensor(
            ess(inference_data, var_names = ["theta"], method = "tail")["theta"].values
        )

        final_tail_ess_per_second = final_tail_ess / wall_time

        final_rhat = torch.Tensor(
            rhat(inference_data, var_names = ["theta"])['theta'].values
        )

        if "accept" in diagnostics.keys():
            acceptance_rate = torch.stack(diagnostics['accept']).T[:, burnin:].to(torch.float).mean(dim = 1)

        return SamplerResults(
            params = params,
            acceptance_rate = acceptance_rate,
            bulk_ess_per_second = final_bulk_ess_per_second,
            tail_ess_per_second = final_tail_ess_per_second,
            rhat = final_rhat,
            wall_time = wall_time,
            burnin = burnin,
            diagnostics = diagnostics,
        )