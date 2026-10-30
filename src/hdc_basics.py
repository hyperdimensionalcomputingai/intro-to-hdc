"""Try binding and bundling without an embedding model or database."""

import torch
import torchhd

from hdc.core import TorchHDEncoder

encoder = TorchHDEncoder()
h_role = encoder.token_hv("key:name")
h_value = encoder.token_hv("value:Seattle")
h_name = torchhd.bind(h_role, h_value)
h_timezone = encoder.association_hv("timezone", "pacific")
h_record = encoder.bundle([h_name, h_timezone])

# Bipolar binding is self-inverse; raw bundling retains the exact sum.
assert torch.equal(torchhd.bind(h_name, h_role), h_value)
assert torch.equal(h_record - h_timezone, h_name)
assert h_record.dtype == torch.float32
print("Binding and raw-bundle recovery checks passed.")
