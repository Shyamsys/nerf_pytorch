import torch
import numpy as np

def positional_encoding(tensor, encoding_function = 5) -> torch.Tensor:
  encoded = [tensor]
  frequency_bands = None
  frequency_bands = 2.0 ** torch.linspace(
            0.0,
            encoding_function - 1,
            encoding_function,
            dtype=tensor.dtype,
            device=tensor.device,
        )

  for freq in frequency_bands:
    for fun in [torch.sin, torch.cos]:
      encoded.append(fun(tensor*freq))

  return torch.cat(encoded, dim=-1)