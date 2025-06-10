
[Activation-aware Weight Quantization](https://hf.co/papers/2306.00978) (AWQ) preserves a small fraction of the weights that are important for LLM performance to compress a model to 4-bits with minimal performance degradation.

There are several libraries for quantizing models with the AWQ algorithm, such as `llm-awq`, `autoawq` or `optimum-intel`. Transformers supports loading models quantized with the `llm-awq` and autoawq libraries. This guide will show you how to load models quantized with autoawq, but the process is similar for `llm-awq` quantized models.

Identify an AWQ-quantized model by checking the `quant_method` key in the models `config.json` file.

```py
{
  "_name_or_path": "/workspace/process/huggingfaceh4_zephyr-7b-alpha/source",
  "architectures": [
    "MistralForCausalLM"
  ],
  ...
  ...
  ...
  "quantization_config": {
    "quant_method": "awq",
    "zero_point": true,
    "group_size": 128,
    "bits": 4,
    "version": "gemm"
  }
}
```

Load the AWQ-quantized model with `from_pretrained()`. This automatically sets the other weights to `fp16` by default for **performance reasons**. Use the `torch_dtype` parameter to load these other weights in a different format.

If the model is loaded on the CPU, use the device_map parameter to move it to a GPU.

Use `attn_implementation` to enable `FlashAttention2` to further accelerate inference

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
  "TheBloke/zephyr-7B-alpha-AWQ",
  attn_implementation="flash_attention_2",
  device_map="cuda:0"
)
```
### Installation

```
pip install autoawq
```

### Fused model
Fused modules offer improved accuracy and performance. They are supported out-of-the-box for AWQ modules for Llama and Mistral architectures, but you can also fuse AWQ modules for unsupported architectures.

> Fused modules **cannot** be combined with other optimization techniques such as FlashAttention2.

Create an `AwqConfig` and set the parameters `fuse_max_seq_len` and `do_fuse=True` to enable fused modules. The `fuse_max_seq_len` parameter is the total sequence length and it should include the context length and the expected generation length. Set it to a larger value to be safe.

```py
import torch
from transformers import AwqConfig, AutoModelForCausalLM

quantization_config = AwqConfig(
    bits=4,
    fuse_max_seq_len=512,
    do_fuse=True,
)
model = AutoModelForCausalLM.from_pretrained(
  "TheBloke/Mistral-7B-OpenOrca-AWQ",
  quantization_config=quantization_config
).to(0)
```

#### Compare fused and un-fused

<!-- ![Fused Forward Memory Plot](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/quantization/fused_forward_memory_plot.png)

![Fused Generate Throughput Plot](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/quantization/fused_generate_throughput_plot.png) -->

<table>
  <tr>
    <td>
      <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/quantization/fused_forward_memory_plot.png" alt="Fused Forward Memory Plot" width="400"/>
    </td>
    <td>
      <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/quantization/fused_generate_throughput_plot.png" alt="Fused Generate Throughput Plot" width="400"/>
    </td>
  </tr>
</table>