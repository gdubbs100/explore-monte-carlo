import torch
import torch.distributions as dist

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

## create base class that defines run and other logging
## sub classes inherit and define step, called in run
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

    @abstractmethod
    def step(self, state: SamplerState) -> tuple[SamplerState, dict]:
        """Step all chains forward one iteration. Return (new_state, info_dict)."""
        ...

    def init_state(self, num_chains: int) -> SamplerState:
        params = self.init_dist.sample((num_chains,)).reshape(num_chains, self.dim)
        return SamplerState(params=params, log_prob=self.log_prob_fn(params))

    def run(self, num_iters: int, num_chains: int, burnin: int = 0):
        assert burnin < num_iters, f"burnin (got {burnin}) must be less than num_iters (got {num_iters})"
        state = self.init_state(num_chains)
        history = torch.empty((num_chains, num_iters + 1, self.dim))
        history[:, 0, :] = state.params
        gelman_rubin = torch.empty((num_iters - (burnin + 2), self.dim))
        diagnostics = defaultdict(list)

        for t in range(num_iters):
            state, info = self.step(state)
            history[:, t + 1, :] = state.params
            if t > burnin + 1:
                gelman_rubin[t - (burnin + 2), :] = self.calc_gelman_rubin(
                    results = history[:, burnin + 1:, :], 
                    num_chains = num_chains
                )
            for k, v in info.items():
                diagnostics[k].append(v)
        
        self.history = history
        self.results = history[:, burnin + 1:, :]
        self.diagnostics = {k: torch.stack(v) for k, v in diagnostics.items()}
        self.gelman_rubin = gelman_rubin
        if "accept" in self.diagnostics:
              self.acceptance_rate = self.diagnostics["accept"].float().mean(dim=0)
        return history
    
    def calc_gelman_rubin(self, results: torch.Tensor, num_chains: int) -> torch.Tensor:
          L = results.shape[1]
          chain_means = results.mean(dim=1)
          grand_mean = chain_means.mean(dim=0)
          B = (L / (num_chains - 1)) * (chain_means - grand_mean).pow(2).sum(dim=0)
          W = ((1 / (L - 1)) * (results - chain_means.unsqueeze(1)).pow(2).sum(dim=1)).mean(dim=0)
          var_hat = ((L - 1) / L) * W + (1 / L) * B
          return torch.sqrt(var_hat / W)