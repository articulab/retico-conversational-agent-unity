# Installation

## Pre-requisties

- pyaudio (installation process depends on your OS : [installation documentation](https://pypi.org/project/PyAudio/))
- CUDA Toolkit installed ([all versions here](https://developer.nvidia.com/cuda-toolkit-archive))

## Basic installation (CPU support)

### clone repo

```bash
git clone https://github.com/articulab/retico-conversational-agent
```

### Create and activate your virtual environment

```{warning}
**python 3.11.7** recommended as the installation has only been tested on this version.
```

With conda :

```bash
conda env create -n [env_name] python=3.11.7
conda activate [env_name]
```

### Install package and its dependencies

```bash
pip install .
```

````{note}
After this, you should be able to run the system on your CPU (but it is not recommended as it will be very slow). You can test the system by executing the main file from `src/retico-conversational-agent/` :

```bash
python main.py
```

````

## Installation to execute system on GPU (with CUDA support)

As many modules dialogue tasks (ASR, NLG, TTS, etc) are fullfilled by Deep Learning models, that needs high computing power to run on a human-like dialogue time-scale (<1 second..), it is highly recommended to execute the system using GPUs. In order to do that you will need to install few GPU-related dependencies.

### CUDA

A CUDA Toolkit is required before following this cuda support installation process.

If you haven't installed CUDA, to check which version to install you can run the `nvidia-smi` command on the terminal :

```bash
$ nvidia-smi
Wed Nov 21 19:41:32 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.72       Driver Version: 410.72       CUDA Version: 10.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 106...  Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   53C    P0    26W /  N/A |    379MiB /  6078MiB |      2%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1324      G   /usr/lib/xorg/Xorg                           225MiB |
|    0      2844      G   compiz                                       146MiB |
|    0     15550      G   /usr/lib/firefox/firefox                       1MiB |
|    0     19992      G   /usr/lib/firefox/firefox                       1MiB |
|    0     23605      G   /usr/lib/firefox/firefox                       1MiB |
```

On the top right, we can see that the maximum version supported by computer's drivers is `CUDA Version: 10.0`.

If you have installed a CUDA Toolkit, to check your installed version, you can run the following command :

```bash
nvcc --version
```

the command will return something similar if you have cuda toolkit version **12.2** installed :

```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Jun_13_19:42:34_Pacific_Daylight_Time_2023
Cuda compilation tools, release 12.2, V12.2.91
Build cuda_12.2.r12.2/compiler.32965470_0
```

### Install cuda support for Deep Learning retico modules (ASR, LLM, TTS) (to speed up greatly the system's execution)

Modify the following line with regards to your installed cuda toolkit version (Here `cu118` is corresponds to an installed cuda 11.8 version).

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall --no-cache
```

To check that your installation worked, test that torch has cuda supported :

```bash
python
import torch
torch.cuda.is_available()
>>> True
```

If `torch.cuda.is_available()` returns `True`, it worked and your DL models should be able to run on GPU. If it returns `False` your torch or cuda installation has problems (maybe the cuda and torch versions do not correspond to one another).

````{note}
If it returns `True`, you can test the system's execution on GPU (it should speed up the execution greatly, as the system should be able to answer in less than 3 seconds) by executing the main file from `src/simple_retico_agent/` :

```bash
python main.py
```

````

### Installation for llama-cpp-python's cuda support (GPU execution)

`llama-ccp-python` is a little bit particular and you will need to reinstall a cuda-supported version to make your LLM run on GPU. Here is the full installation documentation : [https://llama-cpp-python.readthedocs.io/en/latest/](https://llama-cpp-python.readthedocs.io/en/latest/)

To reinstall llama-cpp-python with cuda support on Windows :

```bash
pip uninstall llama-cpp-python && set "CMAKE_ARGS=-DGGML_CUDA=on" && set "FORCE_CMAKE=1" && pip install --no-cache-dir llama-cpp-python 
```

to check that your LLM is running on GPU, run your system with LlmDmModule's verbose argument set to True :

```python
llm = LlmDmModule(
    model_path=None,
    model_repo=model_repo,
    model_name=model_name,
    dialogue_history=dialogue_history,
    device=device,
    context_size=context_size,
    verbose=True,
)
```

And look for the `llm_load_tensors:` lines outputted in terminal during modules initialization :

The following lines are from an execution which has the LLM running on GPU, we can see that the complete model (33 out of 33 NN layers) has been offloaded to GPU :

```bash
...
llm_load_tensors: offloading 32 repeating layers to GPU
llm_load_tensors: offloading output layer to GPU
llm_load_tensors: offloaded 33/33 layers to GPU
llm_load_tensors:        CUDA0 model buffer size =  4095.05 MiB
llm_load_tensors:   CPU_Mapped model buffer size =    70.31 MiB
...
```

The following lines are from an execution which has the LLM running on CPU (no mention of GPU or CUDA, and the `CPU_Mapped model buffer` contains the full model):

```bash
...
llm_load_tensors:   CPU_Mapped model buffer size =  4165.37 MiB
...
```

If it works, you will experience an even faster system than previously when you installed the cuda support, as the LLM inference should be very quick (first clause delivered in less than 500ms)

````{note}
If it doesn't work (many people at Articulab had a lot of troubles installing llama-cpp-python), you can try to reinstall it a second time with the same command (it works sometimes). You could also have a final solution, if you are using conda, you can try to copy an exact conda environement that was able to run the system on GPU with llama-cpp-python cuda support. After the env installed, try to reinstall llama-cpp-python's cuda supported version:

```bash
conda env update -n [env_name] -f env_requirements/retico_cuda_curr.yml --prune
pip uninstall llama-cpp-python && set "CMAKE_ARGS=-DGGML_CUDA=on" && set "FORCE_CMAKE=1" && pip install llama-cpp-python --no-cache-dir
```

````
