import torch
from transformers.models.auto import AutoModelForCausalLM, AutoConfig
import coremltools as ct
import numpy as np
from typing import Any, Optional, Sequence
from transformers.cache_utils import Cache
from argparse import ArgumentParser
import subprocess
from huggingface_hub import (
    create_repo,
    upload_file,
    hf_hub_download,
    upload_folder,
    whoami,
    ModelCard,
    ModelCardData,
)
import os


def log(text):
    print(f"\033[92m\033[1m{text}\033[0m")


parser = ArgumentParser(
    prog="CausalLM.py", description="Convert a Causal LM to Core ML model"
)
parser.add_argument(
    "--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model ID"
)
parser.add_argument(
    "--4-bit", action="store_true", help="Linear quantize model to 4 bits"
)
parser.add_argument(
    "--8-bit", action="store_true", help="Linear quantize model to 8 bits"
)
parser.add_argument(
    "--half", action="store_true", help="Load the model in half precision"
)
parser.add_argument(
    "--compile", action="store_true", help="Save the model as a .mlmodelc"
)
parser.add_argument("--context-size", type=int, default=2048, help="Context size")
parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
parser.add_argument(
    "--upload", action="store_true", help="Upload the model to Hugging Face"
)
args = parser.parse_args()


class KVCache(Cache):
    def __init__(
        self,
        *,
        shape: Sequence[int],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.past_seen_tokens: int = 0
        self.k: torch.Tensor = torch.zeros(shape, dtype=dtype)
        self.v: torch.Tensor = torch.zeros(shape, dtype=dtype)

    def update(
        self,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position = cache_kwargs.get("cache_position", None)
        assert position is not None, "cache_position required to update cache."

        new_tokens = position.shape[-1]
        context_size = self.k.shape[3]

        begin, end = context_size - self.past_seen_tokens, context_size

        self.k[
            layer_idx, :, : k_state.shape[1], begin - new_tokens : end - new_tokens, :
        ] = self.k[layer_idx, :, : k_state.shape[1], begin:end, :]

        self.v[
            layer_idx, :, : v_state.shape[1], begin - new_tokens : end - new_tokens, :
        ] = self.v[layer_idx, :, : v_state.shape[1], begin:end, :]

        begin = end - new_tokens

        self.k[layer_idx, :, : k_state.shape[1], begin:end, :] = k_state
        self.v[layer_idx, :, : v_state.shape[1], begin:end, :] = v_state

        begin = end - self.past_seen_tokens - new_tokens

        k_state = self.k[layer_idx, :, :, begin:end, :]
        v_state = self.v[layer_idx, :, :, begin:end, :]

        return k_state, v_state

    def get_seq_length(self, _: int = 0) -> int:
        return self.past_seen_tokens


class Model(torch.nn.Module):
    def __init__(
        self,
        model_path: str,
        *,
        batch_size: int = 1,
        context_size: int = 4096,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        config: AutoConfig = self.model.config
        self.kv_cache_shape: tuple[int, ...] = (
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            context_size,
            config.hidden_size // config.num_attention_heads,
        )
        self.kv_cache = KVCache(shape=self.kv_cache_shape, dtype=dtype)
        self.register_buffer("key_cache", self.kv_cache.k)
        self.register_buffer("value_cache", self.kv_cache.v)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.kv_cache.past_seen_tokens = causal_mask.shape[-1] - input_ids.shape[-1]
        return self.model(
            input_ids,
            attention_mask=causal_mask,
            past_key_values=self.kv_cache,
            use_cache=True,
        ).logits


log("Loading model…")
model: torch.nn.Module = Model(
    args.model,
    batch_size=args.batch_size,
    context_size=args.context_size,
    dtype=torch.float16 if args.half else torch.float32,
).eval()

log("Tracing model…")
example_inputs: tuple[torch.Tensor, ...] = (
    torch.zeros((1, 2), dtype=torch.int32),
    torch.zeros((1, 1, 2, 2), dtype=torch.float16 if args.half else torch.float32),
)

traced_model: torch.jit.ScriptModule = torch.jit.trace(
    model.eval(), example_inputs=example_inputs
)

log("Converting model…")
query_size = ct.RangeDim(lower_bound=1, upper_bound=args.context_size, default=1)
final_step = ct.RangeDim(lower_bound=1, upper_bound=args.context_size, default=1)
inputs: list[ct.TensorType] = [
    ct.TensorType(
        shape=(args.batch_size, query_size), dtype=np.int32, name="input_ids"
    ),
    ct.TensorType(
        shape=(args.batch_size, 1, query_size, final_step),
        dtype=np.float16,
        name="causal_mask",
    ),
]
states: list[ct.StateType] = [
    ct.StateType(
        wrapped_type=ct.TensorType(shape=model.kv_cache_shape, dtype=np.float16),
        name="key_cache",
    ),
    ct.StateType(
        wrapped_type=ct.TensorType(shape=model.kv_cache_shape, dtype=np.float16),
        name="value_cache",
    ),
]
outputs: list[ct.TensorType] = [ct.TensorType(dtype=np.float16, name="logits")]
mlmodel: ct.models.MLModel = ct.convert(
    traced_model,
    inputs=inputs,
    outputs=outputs,
    states=states,
    minimum_deployment_target=ct.target.macOS15,
)

if vars(args)["4_bit"] or vars(args)["8_bit"]:
    log("Quantizing model…")
    op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int8" if vars(args)["8_bit"] else "int4",
        granularity="per_block",
        block_size=32,
    )
    config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
    mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=config)

log("Saving model…")
model_name = args.model.split("/")[-1]
if vars(args)["4_bit"]:
    model_name = f"{model_name}-4bit"
elif vars(args)["8_bit"]:
    model_name = f"{model_name}-8bit"

file_name = f"models/{model_name}.mlpackage"
mlmodel.save(file_name)

if args.compile or args.upload:
    log("Compiling model…")
    subprocess.run(
        [
            "xcrun",
            "coremlcompiler",
            "compile",
            file_name,
            "models/",
        ]
    )

if args.upload:
    log("Uploading model…")
    user = whoami()
    username = user["name"]
    repo_id = f"{username}/coreml-{model_name}"
    create_repo(repo_id=repo_id, exist_ok=True)
    upload_folder(
        folder_path=f"{os.path.dirname(os.path.realpath(__file__))}/models/{model_name}.mlmodelc",
        path_in_repo=f"{model_name}.mlmodelc",
        repo_id=repo_id,
    )
    upload_file(
        path_or_fileobj=bytes("{}", encoding="utf-8"),
        path_in_repo="config.json",
        repo_id=repo_id,
    )
    upload_file(
        path_or_fileobj=hf_hub_download(repo_id=args.model, filename="tokenizer.json"),
        path_in_repo="tokenizer.json",
        repo_id=repo_id,
    )
    upload_file(
        path_or_fileobj=hf_hub_download(
            repo_id=args.model, filename="tokenizer_config.json"
        ),
        path_in_repo="tokenizer_config.json",
        repo_id=repo_id,
    )
    parent_card = ModelCard.load(args.model).data.to_dict()
    tags: list[str] = parent_card.get("tags", [])
    tags.append("CoreMLPipelines")
    for tag in ["pytorch", "safetensors", "onnx", "transformers.js"]:
        if tag in tags:
            tags.remove(tag)
    card_data = ModelCardData(
        language=parent_card.get("language"),
        library_name="coreml",
        license=parent_card.get("license"),
        license_link=parent_card.get("license_link"),
        pipeline_tag=parent_card.get("pipeline_tag"),
        tags=tags,
    )
    card_content = f"""
---
{ card_data.to_yaml() }
---

# coreml-{model_name}

This model was converted from [{args.model}](https://hf.co/{args.model}) to CoreML using [coremlpipelinestools](https://github.com/finnvoor/CoreMLPipelines/tree/main/coremlpipelinestools).

### Use with [CoreMLPipelines](https://github.com/finnvoor/CoreMLPipelines)

```swift
import CoreMLPipelines

let pipeline = try await TextGenerationPipeline(
    modelName: "{repo_id}"
)
let stream = pipeline(
    messages: [[
        "role": "user",
        "content": "Write a poem about Ireland"
    ]]
)
for try await text in stream {{
    print(text, terminator: "")
    fflush(stdout)
}}
print("")
```
    """
    card = ModelCard(card_content)
    card.push_to_hub(repo_id=repo_id)

    log(f"Model uploaded to https://hf.co/{repo_id}")

log("Done!")
