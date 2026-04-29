#!/usr/bin/env bash
# RunPod-side setup. Run once after `git clone` on the pod.
# Assumes a stock pytorch container (torch + transformers preinstalled);
# adds einops and (optionally) mamba_ssm fast-path kernels.
set -euo pipefail

echo "[*] python: $(python3 --version)"
echo "[*] torch:  $(python3 -c 'import torch; print(torch.__version__, "cuda=", torch.cuda.is_available())')"

pip install --no-input einops transformers "numpy<2" matplotlib

# Mamba fast-path kernels. Skip if it fails -- transformers will fall back
# to the slow sequential implementation.
pip install --no-input mamba-ssm causal-conv1d || \
    echo "[!] mamba-ssm / causal-conv1d install failed -- will use slow path"

echo "[*] sanity:"
python3 -c "
import torch, transformers, einops
print('  torch     ', torch.__version__)
print('  cuda OK   ', torch.cuda.is_available())
print('  device    ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print('  trans     ', transformers.__version__)
print('  einops    ', einops.__version__)
"
echo "[*] ready. Try:  python3 exp_a_mamba/a2_pretrained_mamba.py"
