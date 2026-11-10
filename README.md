# Fine-tune GPT-2

Local LoRA fine-tune of [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) on TinyStories, for Apple Silicon.

The PyTorch weights live in `models/gpt2`. TensorFlow, JAX, and ONNX copies are not downloaded.

```bash
uv run gpt2-finetune
```

A longer run:

```bash
uv run gpt2-finetune --max-steps 500 --dataset-split "train[:8000]"
```

The adapter is written to `outputs/gpt2-tinystories-lora`.
