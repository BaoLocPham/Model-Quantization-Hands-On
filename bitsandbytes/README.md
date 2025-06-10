## bitandbytes

Original content from Huggingface [documentation](https://huggingface.co/docs/transformers/v4.52.3/quantization/bitsandbytes).
- `Quantized Linear Layers`: Linear8bitLt and Linear4bit layers that replace standard PyTorch linear layers with memory-efficient quantized alternatives
- `Optimized Optimizers`: 8-bit versions of common optimizers through its optim module, enabling training of large models with reduced memory requirements
- `Matrix Multiplication`: Optimized matrix multiplication operations that leverage the quantized format

bitandbytes quantizer's space: [link](https://huggingface.co/spaces/bnb-community/bnb-my-repo)

### Installation
```
pip install --upgrade transformers accelerate bitsandbytes
```

### Sample code for quantization

Quantizing a model in 8-bit halves the memory-usage, and for large models, set `device_map="auto"` to efficiently distribute the weights across all available GPUs.

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model_8bit = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-1b7",
    device_map="auto",
    quantization_config=quantization_config
)
```

Change the `load_in_8bit=True` to `load_in_4bit=True` to quantizate model to 4bit

Once a model is quantized to 8-bit, you can’t push the quantized weights to the Hub unless you’re using the latest version of Transformers and bitsandbytes. If you have the latest versions, then you can push the 8-bit model to the Hub with push_to_hub(). The quantization config.json file is pushed first, followed by the quantized model weights.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-560m", 
    device_map="auto",
    quantization_config=quantization_config
)

model.push_to_hub("bloom-560m-8bit")
```

-> Load quantized models with `from_pretrained()`  without a `quantization_config`.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{your_username}/bloom-560m-8bit", device_map="auto")
```

### Offloading
8-bit models can offload weights between the CPU and GPU to fit very large models into memory. The weights dispatched to the CPU are stored in float32 and aren’t converted to 8-bit. For example, enable offloading for bigscience/bloom-1b7 through BitsAndBytesConfig
```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
```
Design a custom device map to fit everything on your GPU except for the `lm_head`, which is dispatched to the CPU.

```py
device_map = {
    "transformer.word_embeddings": 0,
    "transformer.word_embeddings_layernorm": 0,
    "lm_head": "cpu",
    "transformer.h": 0,
    "transformer.ln_f": 0,
}
```
Now load your model with the custom device_map and `quantization_config`.
```py
model_8bit = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-1b7",
    torch_dtype="auto",
    device_map=device_map,
    quantization_config=quantization_config,
)
```

### Skip module conversion
For some models, like [Jukebox](https://huggingface.co/docs/transformers/en/model_doc/jukebox), you don’t need to quantize every module to 8-bit because it can actually cause instability. With [Jukebox](https://huggingface.co/docs/transformers/en/model_doc/jukebox), there are several lm_head modules that should be skipped using the `llm_int8_skip_modules` parameter in `BitsAndBytesConfig`.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "bigscience/bloom-1b7"

quantization_config = BitsAndBytesConfig(
    llm_int8_skip_modules=["lm_head"],
)

model_8bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    quantization_config=quantization_config,
)
```

### Finetuning

The [PEFT](https://github.com/huggingface/peft) library supports fine-tuning large models like `flan-t5-large` and `facebook/opt-6.7b` with 8-bit quantization. You don’t need to pass the device_map parameter for training because it automatically loads your model on a GPU. However, you can still customize the device map with the `device_map` parameter (device_map="auto" should only be used for inference).