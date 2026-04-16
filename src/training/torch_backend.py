"""PyTorch/PEFT backend for Colab and CUDA environments."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from src.maze_repr import SYSTEM_PROMPT
from src.training.backend import MazePolicy
from src.training.config import RLConfig, SFTConfig, resolve_model_for_backend, resolve_records
from src.training.rewards import RewardFn


def torch_device() -> str:
    """Return the preferred torch device string."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class SupervisedCollator:
    """Pad causal-LM SFT batches while preserving masked labels."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.tokenizer.pad_token_id] * pad)
            attention_mask.append(feature["attention_mask"] + [0] * pad)
            labels.append(feature["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class TorchPolicy:
    """Torch-backed generation wrapper."""

    backend_name = "torch"

    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_completion(self, record, max_tokens: int = 32, temperature: float = 0.0) -> str:
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": record.maze_str},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        do_sample = temperature > 0
        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5),
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        completion_ids = generated[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


class TorchBackend:
    """PyTorch/PEFT implementation of the SFT -> RL workflow."""

    name = "torch"

    def __init__(self):
        self.device = torch_device()
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

    def _load_tokenizer(self, model_id: str):
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _base_model_kwargs(self):
        return {
            "torch_dtype": self.dtype,
            "trust_remote_code": True,
        }

    def _load_base_model(self, model_id: str):
        model = AutoModelForCausalLM.from_pretrained(model_id, **self._base_model_kwargs())
        model.to(self.device)
        return model

    def _lora_config(self, rank: int) -> LoraConfig:
        return LoraConfig(
            r=rank,
            lora_alpha=rank,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    def _prompt_text(self, tokenizer, record) -> str:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": record.maze_str},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _prompt_plus_answer_text(self, tokenizer, record) -> str:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": record.maze_str},
                {"role": "assistant", "content": record.solution_moves},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

    def _build_sft_dataset(self, tokenizer, records) -> Dataset:
        rows = []
        for record in records:
            prompt = self._prompt_text(tokenizer, record)
            full_text = self._prompt_plus_answer_text(tokenizer, record) + tokenizer.eos_token
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
            rows.append(
                {
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels,
                }
            )
        return Dataset.from_list(rows)

    def _make_training_args(
        self,
        output_dir: Path,
        config: SFTConfig,
        has_eval: bool,
    ) -> TrainingArguments:
        """Build TrainingArguments compatibly across transformers versions."""
        kwargs = {
            "output_dir": str(output_dir / "workdir"),
            "num_train_epochs": config.epochs if config.epochs is not None else 1.0,
            "max_steps": config.iters if config.epochs is None else -1,
            "per_device_train_batch_size": config.batch_size,
            "per_device_eval_batch_size": config.batch_size,
            "learning_rate": config.lr,
            "weight_decay": 0.0,
            "logging_steps": 10,
            "save_strategy": "no",
            "report_to": [],
            "remove_unused_columns": False,
            "fp16": self.device == "cuda",
            "gradient_checkpointing": self.device == "cuda",
        }

        if has_eval:
            strategy_name = "evaluation_strategy"
            if strategy_name not in inspect.signature(TrainingArguments.__init__).parameters:
                strategy_name = "eval_strategy"
            kwargs[strategy_name] = "steps" if config.epochs is None else "epoch"
            kwargs["eval_steps"] = 50 if config.epochs is None else None
        else:
            strategy_name = "evaluation_strategy"
            if strategy_name not in inspect.signature(TrainingArguments.__init__).parameters:
                strategy_name = "eval_strategy"
            kwargs[strategy_name] = "no"

        return TrainingArguments(**kwargs)

    def train_sft(self, config: SFTConfig) -> Path:
        model_id = resolve_model_for_backend(config.model, self.name)
        tokenizer = self._load_tokenizer(model_id)
        train_records = resolve_records(config.dataset, config.records)
        val_records = (
            resolve_records(config.val_dataset, config.val_records)
            if config.val_dataset is not None or config.val_records is not None
            else None
        )

        model = self._load_base_model(model_id)
        model.config.use_cache = False
        model = get_peft_model(model, self._lora_config(config.lora_rank))

        train_ds = self._build_sft_dataset(tokenizer, train_records)
        val_ds = self._build_sft_dataset(tokenizer, val_records) if val_records else None
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = self._make_training_args(
            output_dir=output_dir,
            config=config,
            has_eval=val_ds is not None,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=SupervisedCollator(tokenizer),
        )
        trainer.train()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        return output_dir

    def _load_policy_model(self, model_id: str, adapter_dir: str | Path, trainable: bool):
        base = self._load_base_model(model_id)
        model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=trainable)
        model.to(self.device)
        model.config.use_cache = False if trainable else True
        if trainable:
            model.train()
        else:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        return model

    def _encode_prompt(self, tokenizer, record):
        prompt = self._prompt_text(tokenizer, record)
        return tokenizer(prompt, return_tensors="pt").to(self.device)

    def _sample_group(
        self,
        model,
        tokenizer,
        record,
        group_size: int,
        temperature: float,
        max_tokens: int,
        reward_fn: RewardFn,
    ):
        inputs = self._encode_prompt(tokenizer, record)
        generated = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=group_size,
        )
        prompt_len = inputs["input_ids"].shape[1]
        samples = []
        maze = record.to_maze()
        for seq in generated:
            completion_ids = seq[prompt_len:]
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            samples.append(
                {
                    "completion_ids": completion_ids.tolist(),
                    "completion": completion_text,
                    "reward": reward_fn(completion_text, maze),
                }
            )
        return samples

    def _sequence_logprob(self, model, prompt_ids: list[int], completion_ids: list[int]) -> torch.Tensor:
        if len(completion_ids) == 0:
            return torch.tensor(0.0, device=self.device)

        full_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(full_ids)
        logits = model(input_ids=full_ids, attention_mask=attention_mask).logits[:, :-1, :]
        log_probs = torch.log_softmax(logits, dim=-1)

        start = len(prompt_ids) - 1
        end = start + len(completion_ids)
        target = torch.tensor(completion_ids, dtype=torch.long, device=self.device).view(1, -1, 1)
        token_log_probs = log_probs[:, start:end, :].gather(2, target).squeeze(-1)
        return token_log_probs.sum()

    def train_rl(self, config: RLConfig, reward_fn: RewardFn) -> Path:
        model_id = resolve_model_for_backend(config.model, self.name)
        if config.adapters is None:
            raise ValueError("Torch RL requires an SFT adapter path via config.adapters")

        tokenizer = self._load_tokenizer(model_id)
        policy_model = self._load_policy_model(model_id, config.adapters, trainable=True)
        ref_model = self._load_policy_model(model_id, config.adapters, trainable=False)
        optimizer = torch.optim.AdamW(
            (p for p in policy_model.parameters() if p.requires_grad),
            lr=config.lr,
        )
        records = resolve_records(config.dataset, config.records)
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logs = []
        for step in range(1, config.max_steps + 1):
            record = records[(step - 1) % len(records)]
            policy_model.eval()
            samples = self._sample_group(
                policy_model,
                tokenizer,
                record,
                config.num_generations,
                config.temperature,
                config.max_tokens,
                reward_fn,
            )
            rewards = torch.tensor(
                [sample["reward"] for sample in samples],
                dtype=torch.float32,
                device=self.device,
            )
            advantages = (rewards - rewards.mean()) / rewards.std().clamp_min(1e-6)
            prompt_ids = self._encode_prompt(tokenizer, record)["input_ids"][0].tolist()

            policy_model.train()
            optimizer.zero_grad()
            losses = []
            for advantage, sample in zip(advantages, samples):
                completion_ids = sample["completion_ids"]
                log_prob = self._sequence_logprob(policy_model, prompt_ids, completion_ids)
                with torch.no_grad():
                    ref_lp = self._sequence_logprob(ref_model, prompt_ids, completion_ids)
                losses.append(-(advantage.detach() * log_prob) + config.beta * (log_prob - ref_lp))

            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy_model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()

            best_idx = max(range(len(samples)), key=lambda idx: samples[idx]["reward"])
            metrics = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "reward_mean": float(rewards.mean().detach().cpu()),
                "reward_std": float(rewards.std().detach().cpu()),
                "reward_max": float(rewards.max().detach().cpu()),
                "best_reward": samples[best_idx]["reward"],
                "best_completion": samples[best_idx]["completion"],
            }
            logs.append(metrics)
            if step == 1 or step % config.log_interval == 0:
                print(
                    f"  Step {step:4d}/{config.max_steps} | "
                    f"loss={metrics['loss']:7.3f} | "
                    f"reward={metrics['reward_mean']:6.3f} ± {metrics['reward_std']:.3f} | "
                    f"max={metrics['reward_max']:6.3f}"
                )

        policy_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        with open(output_dir / "log.json", "w") as handle:
            json.dump(logs, handle, indent=2)
        return output_dir

    def load_policy(
        self,
        model: str | None = None,
        adapter_path: str | Path | None = None,
        lora_rank: int = 16,
    ) -> MazePolicy:
        del lora_rank
        model_id = resolve_model_for_backend(model, self.name)
        tokenizer = self._load_tokenizer(model_id)
        if adapter_path is None:
            model_obj = self._load_base_model(model_id)
            model_obj.eval()
        else:
            model_obj = self._load_policy_model(model_id, adapter_path, trainable=False)
        return TorchPolicy(model_obj, tokenizer, self.device)
