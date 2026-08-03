## Walking videos

The code is open-source and free to use. It is aimed for, but not limited to, academic research. We welcome forking of this repository, pull requests, and any contributions in the spirit of open science and open-source code. For inquiries about collaboration, you may contact Md Shadab Alam (md_shadab_alam@outlook.com) or Pavlo Bazilinskyy (pavlo.bazilinskyy@gmail.com).

## Getting started
[![Python Version](https://img.shields.io/badge/python-3.12.13-blue.svg)](https://www.python.org/downloads/release/python-31213/)
[![Package Manager: uv](https://img.shields.io/badge/package%20manager-uv-green)](https://docs.astral.sh/uv/)

Tested with **Python 3.12.13** and the [`uv`](https://docs.astral.sh/uv/) package manager.  
Follow these steps to set up the project.

**Step 1:** Install `uv`. `uv` is a fast Python package and environment manager. Install it using one of the following methods:

**macOS / Linux (bash/zsh):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**Alternative (if you already have Python and pip):**
```bash
pip install uv
```

**Step 2:** Fix permissions (if needed):t

Sometimes `uv` needs to create a folder under `~/.local/share/uv/python` (macOS/Linux) or `%LOCALAPPDATA%\uv\python` (Windows).  
If this folder was created by another tool (e.g. `sudo`), you may see an error like:
```lua
error: failed to create directory ... Permission denied (os error 13)
```

To fix it, ensure you own the directory:

### macOS / Linux
```bash
mkdir -p ~/.local/share/uv
chown -R "$(id -un)":"$(id -gn)" ~/.local/share/uv
chmod -R u+rwX ~/.local/share/uv
```

### Windows
```powershell
# Create directory if it doesn't exist
New-Item -ItemType Directory -Force "$env:LOCALAPPDATA\uv"

# Ensure you (the current user) own it
# (usually not needed, but if permissions are broken)
icacls "$env:LOCALAPPDATA\uv" /grant "$($env:UserName):(OI)(CI)F"
```

**Step 3:** After installing, verify:
```bash
uv --version
```

**Step 4:** Clone the repository:
```command line
git clone https://github.com/Shaadalam9/ASMR-analysis
cd multiped
```

**Step 5:** Ensure correct Python version. If you don’t already have Python 3.12.13 installed, let `uv` fetch it:
```command line
uv python install 3.12.13
```
The repo should contain a .python-version file so `uv` will automatically use this version.

**Step 6:** Create and sync the virtual environment. This will create **.venv** in the project folder and install dependencies exactly as locked in **uv.lock**:
```command line
uv sync --frozen
```

**Step 7:** Activate the virtual environment:

**macOS / Linux (bash/zsh):**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (cmd.exe):**
```bat
.\.venv\Scripts\activate.bat
```

**Step 8:** Ensure that dataset are present. Place required datasets (including **mapping.csv**) into the **data/** directory:


**Step 9:** Run the code:
```command line
python3 main.py
```

## Spike 1 and Run:ai execution

The project now validates the GPUs and persistent storage visible inside the
Run:ai container before loading either model.

Use `spike1.env.example` for automatic handling. Automatic mode selects:

```text
1 visible GPU  -> sequential Qwen then InternVL on cuda:0
2 visible GPUs -> Qwen on cuda:0 and InternVL on cuda:1
```

For an explicit configuration, copy one of these files:

```bash
cp spike1-one-gpu.env.example spike1.env
# or
cp spike1-two-gpu.env.example spike1.env
```

Replace the example DDN path with the path mounted by Run:ai. Load the file in
Bash and validate the environment before running the pipeline:

```bash
set -a
source spike1.env
set +a
python check_spike1.py
python main.py
```

The Run:ai interface can inject the same values directly, in which case the
`source` step is unnecessary.

Important environment values:

```text
SPIKE1_MODE=1
WALK_DATA_DIR=/absolute/path/to/the/ddn/mount
PIPELINE_MODE=auto
SEQUENTIAL_DEVICE=cuda:0
TEXT_DEVICE=cuda:0
VLM_DEVICE=cuda:1
TEXT_LLM_4BIT=0
INTERNVL_4BIT=0
```

Run:ai remaps allocated GPUs to logical device numbers inside the container.
Use `cuda:0` for the first allocated GPU and `cuda:1` for the second allocated
GPU. Do not use the physical host GPU number.

Boolean settings accept `1`, `0`, `true`, `false`, `yes`, `no`, `on`, and
`off`. Limit settings such as `MAX_VIDEOS_PER_RUN` also accept `none` or
`unlimited`.
