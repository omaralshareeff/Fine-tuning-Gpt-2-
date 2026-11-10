import argparse
import os
from dataclasses import dataclass

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, set_seed
from trl import SFTConfig, SFTTrainer

MODEL_PATH = "models/gpt2"
DATASET_ID = "roneneldan/TinyStories"
OUTPUT_DIR = "outputs/gpt2-tinystories-lora"
PROMPT = "Once upon a time there was a little girl named Lucy."
MIN_STORY_CHARS = 40


@dataclass(frozen=True)
class TrainSettings:
    model_path: str = MODEL_PATH
    dataset_split: str = "train[:2000]"
    output_dir: str = OUTPUT_DIR
    max_length: int = 256
    max_steps: int = 100
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    seed: int = 42


def is_trainable_story(text: object, min_chars: int = MIN_STORY_CHARS) -> bool:
    return isinstance(text, str) and len(text.strip()) >= min_chars


def load_stories(split: str) -> Dataset:
    stories = load_dataset(DATASET_ID, split=split)
    stories = stories.filter(lambda row: is_trainable_story(row.get("text")))
    columns = [name for name in stories.column_names if name != "text"]
    return stories.remove_columns(columns)


def training_dtype() -> torch.dtype:
    if torch.backends.mps.is_available():
        return torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def gpt2_lora() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj", "c_fc"],
        fan_in_fan_out=True,
    )


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_causal_lm(model_path: str, dtype: torch.dtype) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    return model.to(device)


def sample_completion(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    seed: int,
) -> str:
    set_seed(seed)
    model.eval()
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def train(settings: TrainSettings) -> str:
    set_seed(settings.seed)
    dtype = training_dtype()
    print(f"Loading {DATASET_ID} {settings.dataset_split}", flush=True)
    stories = load_stories(settings.dataset_split)
    split = stories.train_test_split(test_size=0.05, seed=settings.seed)
    print(f"Train rows: {len(split['train'])}  Eval rows: {len(split['test'])}", flush=True)
    use_bf16 = dtype == torch.bfloat16
    config = SFTConfig(
        output_dir=settings.output_dir,
        max_steps=settings.max_steps,
        max_length=settings.max_length,
        per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=settings.batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        learning_rate=settings.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=settings.max_steps,
        save_total_limit=1,
        packing=False,
        dataloader_num_workers=0,
        dataset_text_field="text",
        bf16=use_bf16,
        fp16=False,
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        report_to="none",
        seed=settings.seed,
        model_init_kwargs={"dtype": dtype},
    )
    print("Preparing model", flush=True)
    trainer = SFTTrainer(
        model=settings.model_path,
        args=config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        peft_config=gpt2_lora(),
    )
    print(f"Device: {trainer.model.device}  dtype: {trainer.model.dtype}", flush=True)
    print("Starting training", flush=True)
    trainer.train()
    trainer.save_model(settings.output_dir)
    return settings.output_dir


def parse_args() -> TrainSettings:
    parser = argparse.ArgumentParser(description="LoRA fine-tune GPT-2 on TinyStories.")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--dataset-split", default="train[:2000]")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = TrainSettings(
        model_path=args.model_path,
        dataset_split=args.dataset_split,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    tokenizer = load_tokenizer(settings.model_path)
    base = load_causal_lm(settings.model_path, torch.float32)
    print("Base model:\n" + sample_completion(base, tokenizer, PROMPT, settings.seed))
    del base
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    saved = train(settings)
    tuned = load_causal_lm(settings.model_path, torch.float32)
    tuned = PeftModel.from_pretrained(tuned, saved)
    print("Fine-tuned model:\n" + sample_completion(tuned, tokenizer, PROMPT, settings.seed))
    print(f"Adapter saved to {saved}")
