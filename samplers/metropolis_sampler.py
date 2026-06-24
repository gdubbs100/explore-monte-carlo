import torch
from samplers.base_sampler import MCMCSampler, SamplerState

class MetropolisSampler(MCMCSampler):
    def __init__(self, *args, proposal_std: float = 0.45, **kwargs):
        super().__init__(*args, **kwargs)
        self.proposal_std = proposal_std

    def step(self, state: SamplerState) -> tuple[SamplerState, dict]:
        proposal = state.params + self.proposal_std * torch.randn_like(state.params)
        proposal_lp = self.log_prob_fn(proposal)
        log_u = torch.log(torch.rand(state.params.shape[0])) 
        accept = (proposal_lp - state.log_prob) >= log_u

        params = torch.where(accept.unsqueeze(-1), proposal, state.params)
        log_prob = torch.where(accept, proposal_lp, state.log_prob)
        return SamplerState(params, log_prob), {"accept": accept}