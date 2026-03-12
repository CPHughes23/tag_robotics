import torch
import torch.nn as nn
from torch.distributions import Categorical
from rsl_rl.modules.distribution import Distribution


class CategoricalDistribution(Distribution):
    def __init__(self, output_dim: int, **kwargs):
        super().__init__(output_dim)
        self._distribution = None

    def update(self, mlp_output: torch.Tensor) -> None:
        self._distribution = Categorical(logits=mlp_output)

    def sample(self) -> torch.Tensor:
        return self._distribution.sample().unsqueeze(-1).float()

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output.argmax(dim=-1, keepdim=True).float()

    def as_deterministic_output_module(self) -> nn.Module:
        return _ArgmaxOutput()

    @property
    def input_dim(self) -> int:
        return self.output_dim

    @property
    def mean(self) -> torch.Tensor:
        return self._distribution.probs.argmax(dim=-1, keepdim=True).float()

    @property
    def std(self) -> torch.Tensor:
        return self._distribution.entropy().unsqueeze(-1)

    @property
    def entropy(self) -> torch.Tensor:
        return self._distribution.entropy()

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return (self._distribution.logits,)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        if outputs.dim() == 2 and outputs.shape[-1] == self.output_dim:
            # one-hot → index
            outputs = outputs.argmax(dim=-1)
        else:
            outputs = outputs.squeeze(-1).long()
        return self._distribution.log_prob(outputs)

    def kl_divergence(self, old_params, new_params) -> torch.Tensor:
        old_dist = Categorical(logits=old_params[0])
        new_dist = Categorical(logits=new_params[0])
        return torch.distributions.kl_divergence(old_dist, new_dist)

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        pass  # no special init needed for categorical


class _ArgmaxOutput(nn.Module):
    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output.argmax(dim=-1, keepdim=True).float()