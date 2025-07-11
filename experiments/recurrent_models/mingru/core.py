from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from codebase.transformer import InitStdFactor, RMSNorm
from codebase.probe import log_stats

from experiments.recurrent_models.component.rnn_common import conv1d, scan

@dataclass
class BaseMinGRUArgs:
	dim: int = 512
	n_layers: int = 8
	n_heads: int = 1

	multiple_of: int = 256
	ffn_dim_multiplier: Optional[float] = None

	conv_size: Optional[float] = None

	norm_eps: float = 1e-5

	init_base_std: Optional[float] = None
	init_std_factor: str = "disabled"

def sequential_step(
	states: torch.Tensor, a: torch.Tensor, b: torch.Tensor
	) -> torch.Tensor:
	return a * states + b

class GRU(nn.Module):
	def __init__(
		self,
		dim: int,
		hidden_dim: int,
		n_heads: int,
		multiple_of: int,
		ffn_dim_multiplier: Optional[float],
		conv_size: Optional[int] = None,
	):

