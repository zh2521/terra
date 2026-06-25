# Installation

TERRA is published on PyPI as `terra-st` (the import name is `terra`). We
recommend installing with [uv](https://docs.astral.sh/uv/):

```shell
uv pip install terra-st
```

Plain `pip install terra-st` works too.

For a development install from a clone of the [repository](https://github.com/Lotfollahi-lab/terra):

```shell
git clone https://github.com/Lotfollahi-lab/terra.git
cd terra
uv pip install -e ".[dev,test,doc]"
```

## PyTorch / GPU note

TERRA depends on [PyTorch](https://pytorch.org). Install the PyTorch build that
matches your hardware **before** installing TERRA — see the
[official PyTorch install guide](https://pytorch.org/get-started/locally/) for
the exact command for your system.

To pick the right build, check your hardware first:

- **NVIDIA GPU:** run `nvidia-smi` and read the "CUDA Version" shown in the
  top-right, then install the matching CUDA build, e.g.:

  ```shell
  uv pip install torch --index-url https://download.pytorch.org/whl/cu124
  ```

- **CPU only (no GPU):**

  ```shell
  uv pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```

Then install TERRA:

```shell
uv pip install terra-st
```

Verify the install with:

```shell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Optional extras

TERRA ships several optional dependency groups:

| Extra | Install | Purpose |
| --- | --- | --- |
| `hub` | `pip install "terra-st[hub]"` | Publish/download model bundles on the Hugging Face Hub (`terra-hub`). |
| `eval` | `pip install "terra-st[eval]"` | Evaluation utilities (CellPhoneDB, Omnipath). |
| `doc` | `pip install "terra-st[doc]"` | Build the documentation. |
| `test` | `pip install "terra-st[test]"` | Run the test suite. |

## Reproducible environment

For the exact, fully-pinned environment TERRA is developed and tested against, use
the committed lockfile with [uv](https://docs.astral.sh/uv/):

```shell
uv sync
```
