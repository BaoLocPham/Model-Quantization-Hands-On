## GPTQ

The `GPTQModel` and `AutoGPTQ` implements the GPTQ algorithm, a post-training quantization technique where each row of the weight matrix is quantized independently to find a version of the weights that minimizes the error. These weights are quantized to int4, but they’re restored to fp16 on the fly during inference. This can save memory usage by 4x because the int4 weights are dequantized in a fused kernel rather than a GPU’s global memory. Inference is also faster because a lower bitwidth takes less time to communicate.

### Installation

Install Accelerate, Transformers and Optimum first
```
pip install --upgrade accelerate optimum transformers
```

```
pip install gptqmodel --no-build-isolation
```

Create a `GPTQConfig` class and set the number of bits to quantize to, a dataset to calbrate the weights for quantization, and a tokenizer to prepare the dataset.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)
```
You can pass your own dataset as a list of strings, but it is highly recommended to use the same dataset from the GPTQ paper.

```py
dataset = ["auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
gptq_config = GPTQConfig(bits=4, dataset=dataset, tokenizer=tokenizer)
```

Load a model to quantize and pass `GPTQConfig` to `from_pretrained()`. Set `device_map="auto"` to automatically offload the model to a CPU to help fit the model in memory, and allow the model modules to be moved between the CPU and GPU for quantization.

```py
quantized_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", device_map="auto", quantization_config=gptq_config)
```

Reload a quantized model with `from_pretrained()`, and set `device_map="auto"` to automatically distribute the model on all available GPUs to load the model faster without using more memory than needed.
```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq", device_map="auto")
```


### ExLlama

`ExLlama` is a Python/C++/CUDA implementation of the Llama model that is designed for faster inference with 4-bit GPTQ weights (check out these benchmarks). The `ExLlama` kernel is activated by default when you create a GPTQConfig object.

To boost inference speed even further, use the `ExLlamaV2` kernels by configuring the `exllama_config` parameter in `GPTQConfig`.

```py
import torch
from transformers import AutoModelForCausalLM, GPTQConfig

gptq_config = GPTQConfig(bits=4, exllama_config={"version":2})
model = AutoModelForCausalLM.from_pretrained(
    "{your_username}/opt-125m-gptq",
    device_map="auto",
    quantization_config=gptq_config
)
```
