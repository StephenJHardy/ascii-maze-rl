"""
Phase 0 Smoke Test: Verify MLX-Tune works on this Mac.

Tests:
  1. Model loading (downloads on first run)
  2. LoRA adapter creation
  3. Single text generation
  4. Timed batch of 8 generations at 32 tokens (simulating GRPO rollouts)

Run with:
  uv run src/smoke_test.py
"""

import time

DEFAULT_MODEL = "mlx-community/Qwen3.5-0.8B-MLX-4bit"
MAX_SEQ_LENGTH = 512


def test_load_model(model_id: str):
    """Test 1: Load model and tokenizer."""
    print("=" * 60)
    print("Test 1: Loading model and tokenizer")
    print(f"  Model: {model_id}")
    print("=" * 60)

    from mlx_tune import FastLanguageModel

    t0 = time.perf_counter()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    elapsed = time.perf_counter() - t0

    print(f"  Loaded in {elapsed:.1f}s")
    print(f"  Tokenizer vocab size: {tokenizer.vocab_size}")
    print("  PASS\n")
    return model, tokenizer


def test_lora_adapter(model):
    """Test 2: Create LoRA adapter."""
    print("=" * 60)
    print("Test 2: Creating LoRA adapter (r=16)")
    print("=" * 60)

    from mlx_tune import FastLanguageModel

    t0 = time.perf_counter()
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
    )
    elapsed = time.perf_counter() - t0

    print(f"  LoRA adapter created in {elapsed:.1f}s")
    print("  PASS\n")
    return model


def test_single_generation(model, tokenizer):
    """Test 3: Generate a single completion."""
    print("=" * 60)
    print("Test 3: Single text generation")
    print("=" * 60)

    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.7)

    prompt = "What is 2 + 2? Answer with just the number:"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    t0 = time.perf_counter()
    response = mlx_lm.generate(
        model,
        tokenizer,
        prompt=text,
        max_tokens=32,
        sampler=sampler,
    )
    elapsed = time.perf_counter() - t0

    response_tokens = tokenizer.encode(response)
    tps = len(response_tokens) / elapsed if elapsed > 0 else 0

    print(f"  Prompt: {prompt}")
    print(f"  Response: {response}")
    print(f"  Tokens: {len(response_tokens)}")
    print(f"  Time: {elapsed:.2f}s ({tps:.1f} tokens/sec)")
    print("  PASS\n")
    return tps


def test_batch_generation(model, tokenizer):
    """Test 4: Time 8 sequential generations of 32 tokens (simulates GRPO rollouts)."""
    print("=" * 60)
    print("Test 4: 8 × 32-token generations (GRPO rollout simulation)")
    print("=" * 60)

    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.7)

    prompt = (
        "Solve this maze. Output ONLY your sequence of moves (u/d/l/r), space-separated.\n"
        "Maze:\n"
        "# # # # # # #\n"
        "> . . # . . #\n"
        "# # . # . # #\n"
        "# . . . . # #\n"
        "# . # # . . #\n"
        "# . . . # . >\n"
        "# # # # # # #\n"
        "Moves:"
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    responses = []
    total_tokens = 0

    t0 = time.perf_counter()
    for i in range(8):
        response = mlx_lm.generate(
            model,
            tokenizer,
            prompt=text,
            max_tokens=32,
            sampler=sampler,
        )
        responses.append(response)
        total_tokens += len(tokenizer.encode(response))
    elapsed = time.perf_counter() - t0

    tps = total_tokens / elapsed if elapsed > 0 else 0

    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Per generation: {elapsed / 8:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Throughput: {tps:.1f} tokens/sec")
    print("\n  Sample responses (first 3):")
    for i, r in enumerate(responses[:3]):
        print(f"    [{i}] {r[:80]}{'...' if len(r) > 80 else ''}")
    print("  PASS\n")

    return elapsed


def main():
    import sys

    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  ASCII Maze RL — Phase 0 Smoke Test                    ║")
    print("║  Verifying MLX-Tune on this Mac                        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    model, tokenizer = test_load_model(model_id)
    model = test_lora_adapter(model)
    tps = test_single_generation(model, tokenizer)
    batch_time = test_batch_generation(model, tokenizer)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Model: {model_id}")
    print(f"  Generation speed: ~{tps:.0f} tokens/sec")
    print(f"  8 × 32-token rollouts: {batch_time:.1f}s")
    print(f"  Estimated GRPO step time: ~{batch_time * 2:.0f}s")
    print("    (rollouts + backward pass)")
    print()

    if batch_time < 60:
        print("  ✓ Throughput looks viable for GRPO training.")
    elif batch_time < 180:
        print("  △ Throughput is marginal. Consider reducing G to 4.")
    else:
        print("  ✗ Throughput is too slow. Consider smaller model or cloud GPU.")
    print()


if __name__ == "__main__":
    main()
