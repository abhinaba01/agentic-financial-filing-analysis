"""QLoRA fine-tune for numerical reasoning on FinQA (section 5.4, stretch goal).

    Base     Qwen2.5-3B-Instruct (7B only once 3B is comfortable)
    Data     ibm/finqa (trust_remote_code=True)
    Method   QLoRA, 4-bit NF4, LoRA r=16 alpha=32, attention + MLP projections
    T4       batch 1, grad-accum 8-16, max_len 1024, gradient checkpointing, paged AdamW
    Baseline the same base model zero-shot, AND a hosted model on the same split

FinQA supplies multi-step reasoning programs over financial tables, which is the
right supervision for the derived-KPI and reasoning steps. The result worth
reporting is the three-way comparison on identical prompts - base zero-shot vs.
this QLoRA model vs. a hosted frontier model - which ``affa-eval finqa --hosted``
produces.

QLoRA is the cheap checkpointing case: only adapter weights are saved, roughly
50-100MB, so this can checkpoint frequently at negligible cost.

Run::

    python training/train_finqa_qlora.py --output-dir /content/drive/MyDrive/affa/finqa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Both the repo root and src/ must be importable: `training.common` lives at the
# root, `affa` lives under src/. Running this file directly puts only
# training/ on sys.path, so neither resolves without this.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from training.common import (  # noqa: E402
    RunConfig,
    assert_checkpoint_selection_is_valid,
    require_accelerate,
    resolve_checkpoint_dir,
    resume_checkpoint,
    set_global_seed,
    training_arguments,
)

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DATASET = "ibm/finqa"

PROMPT = """Answer the question using the financial data below. Show only the final numeric answer.

{context}

Question: {question}
Answer:"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Adapter dir (put it on Drive)")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--allow-ephemeral", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_accelerate()
    set_global_seed(args.seed)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
    )

    ds = load_dataset(DATASET, trust_remote_code=True)
    train = ds["train"].shuffle(seed=args.seed)
    if args.train_samples:
        train = train.select(range(min(args.train_samples, len(train))))
    validation = ds["validation"] if "validation" in ds else ds["test"]
    validation = validation.shuffle(seed=args.seed).select(
        range(min(args.eval_samples, len(validation)))
    )
    print(f"[data] train={len(train)} validation={len(validation)}")

    assert_checkpoint_selection_is_valid(selection_split="validation", reporting_split="test")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit NF4 with double quantisation: what makes a 3B model trainable in 16GB.
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=quantization, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False  # incompatible with gradient checkpointing

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Attention AND MLP projections, per section 5.4.
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def format_example(row):
        context = "\n".join(
            filter(
                None,
                [
                    " ".join(row.get("pre_text", []) or []),
                    "\n".join(
                        " | ".join(str(c) for c in line) for line in (row.get("table") or [])
                    ),
                    " ".join(row.get("post_text", []) or []),
                ],
            )
        )[:4000]
        prompt = PROMPT.format(context=context, question=row.get("question", ""))
        answer = str(row.get("answer") or row.get("final_result") or "")
        full = f"{prompt} {answer}{tokenizer.eos_token}"

        encoded = tokenizer(full, truncation=True, max_length=args.max_length, padding=False)
        # Mask the prompt: loss is computed on the answer only, so the model is
        # trained to answer rather than to reproduce the table it was given.
        prompt_len = len(
            tokenizer(prompt, truncation=True, max_length=args.max_length)["input_ids"]
        )
        labels = list(encoded["input_ids"])
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        encoded["labels"] = labels
        return encoded

    tokenized_train = train.map(format_example, remove_columns=train.column_names, desc="train")
    tokenized_val = validation.map(
        format_example, remove_columns=validation.column_names, desc="validation"
    )

    output_dir = resolve_checkpoint_dir(args.output_dir, allow_ephemeral=args.allow_ephemeral)
    run_config = RunConfig(
        task="finqa_qlora",
        base_model=args.base_model,
        dataset=DATASET,
        seed=args.seed,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        extra={
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "quantization": "nf4-double",
        },
    )

    trainer = Trainer(
        model=model,
        args=training_arguments(
            output_dir=output_dir,
            # Adapters are ~50-100MB, so frequent saves cost almost nothing.
            save_steps=args.save_steps,
            seed=args.seed,
            metric_for_best_model="loss",
            greater_is_better=False,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            per_device_eval_batch_size=1,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
        ),
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    last = resume_checkpoint(output_dir, run_config)
    trainer.train(resume_from_checkpoint=last)
    model.save_pretrained(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    print(
        "\nThe result worth reporting is the three-way comparison on identical prompts:\n"
        "  1. set models.reasoner.backend='local' and local_adapter to this adapter\n"
        "  2. affa-eval finqa --hosted --limit 200 --output eval_results/finqa.json\n"
        "It measures base zero-shot, this adapter, and the hosted model on one split."
    )
    if args.push_to_hub:
        trainer.push_to_hub()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
