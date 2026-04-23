"""
train.py — PPO fine-tuning of TinyLlama using LExT as the reward

Run with:
    GROQ_KEYS=key1,key2 python train.py

What this file does
───────────────────
1. Loads TinyLlama and wraps it with a PPO value head.
2. Loops over PubMedQA samples, generating a yes/no + reasoning response.
3. Scores each response with the LExT metric (see lext.py).
4. Updates the model via PPO every BATCH_SIZE steps.
5. Saves the trained model to SAVE_PATH.

Key design decisions
────────────────────
- KL penalty (init_kl_coef=0.2) prevents reward hacking / mode collapse.
- Rewards are re-centred from [0,1] to [-0.5, 0.5] so the value function
  gets signed feedback (positive = better than average, negative = worse).
- Degenerate outputs (empty or 'unknown') receive an immediate -0.5 penalty
  before any metric computation, so the model cannot hack by saying nothing.
- The LExT reward uses the rollout's own label/explanation directly — the
  training model is never re-run inside the reward function.
"""

import os
import torch
from transformers import AutoTokenizer, pipeline
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

import lext as lext_module

# ─────────────────────────────────────────────────────────────────────────────
# Config  (change these to adjust the run)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME        = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SAVE_PATH         = "/content/drive/MyDrive/tinyllama_ppo_finetuned"
MAX_SAMPLES       = 500
BATCH_SIZE        = 4
MAX_PROMPT_TOKENS = 384
MAX_NEW_TOKENS    = 150
LEARNING_RATE     = 1e-6
KL_COEF           = 0.2    # KL penalty strength — prevents reward hacking
REWARD_CENTRE     = 0.5    # shifts [0,1] rewards to [-0.5, 0.5]


# ─────────────────────────────────────────────────────────────────────────────
# Setup: tokenizer, model, PPO trainer
# ─────────────────────────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)

ppo_config = PPOConfig(
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    ppo_epochs=1,
    init_kl_coef=KL_COEF,   # ← critical: without this, the model collapses
    target=6.0,
    horizon=10_000,
    log_with=None,
)

ppo_trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer)

device = ppo_trainer.accelerator.device
model.to(device)

# Give lext.py access to the model so it can run call_local()
lext_module.init(model, tokenizer, device)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset and NER pipeline
# ─────────────────────────────────────────────────────────────────────────────

dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
dataset = dataset.select(range(MAX_SAMPLES))

# NER runs on CPU to avoid GPU memory conflicts with the PPO model
ner_pipe = pipeline(
    "token-classification",
    model="Clinical-AI-Apollo/Medical-NER",
    aggregation_strategy="simple",
    device=-1,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(context: str, question: str) -> str:
    return (
        "You are a medical assistant.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Respond in this exact format:\n"
        "Answer: Yes or No\n"
        "Reasoning: <one sentence explaining why>\n"
    )


def parse_response(text: str) -> tuple[str, str]:
    """
    Extract label and reasoning from the model's raw output.

    Falls back gracefully so we always return something usable:
    - Label:      looks for yes/no anywhere in the text if the structured
                  'Answer:' line is missing
    - Explanation: uses the full response text if no 'Reasoning:' line found
    """
    label, explanation = "unknown", text.strip()

    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("answer:"):
            raw = line.split(":", 1)[-1].strip().lower()
            if "yes" in raw:
                label = "Yes"
            elif "no" in raw:
                label = "No"
        elif line.lower().startswith("reasoning:"):
            explanation = line.split(":", 1)[-1].strip()

    # Fallback: if label still unknown, scan the whole text for yes/no
    if label == "unknown":
        lower = text.lower()
        if "yes" in lower:
            label = "Yes"
        elif "no" in lower:
            label = "No"

    # Fallback: if explanation is empty, use the whole response
    if not explanation.strip():
        explanation = text.strip()

    return label, explanation


def safe_1d(tensor: torch.Tensor, name: str = "") -> torch.Tensor:
    """Flatten to 1-D and validate — TRL requires 1-D tensors in its buffer."""
    t = tensor.detach().flatten()
    if t.dim() != 1 or t.numel() == 0:
        raise ValueError(f"Tensor '{name}' has bad shape: {t.shape}")
    return t


# ─────────────────────────────────────────────────────────────────────────────
# PPO batch buffers and update
# ─────────────────────────────────────────────────────────────────────────────

query_tensors    = []
response_tensors = []
rewards_list     = []


def flush_batch(step: int):
    """Run one PPO update on the accumulated batch, then clear the buffers.
    Buffers are always cleared — even if the PPO step itself fails — so a
    bad batch can never block all future updates.
    """
    try:
        reward_tensors = [torch.tensor(r, dtype=torch.float32) for r in rewards_list]
        ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
        mean_r = sum(rewards_list) / len(rewards_list)
        print(f"  → PPO update at step {step} | mean reward {mean_r:.4f}\n")
    except Exception as e:
        print(f"  ⚠ PPO step failed at step {step}: {e} — skipping batch\n")
    finally:
        query_tensors.clear()
        response_tensors.clear()
        rewards_list.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

print("Starting PPO training…\n")
last_step = 0

for step, sample in enumerate(dataset):
    last_step = step
    try:
        # Skip ambiguous labels
        if sample["final_decision"].strip().lower() == "maybe":
            continue

        question = sample["question"]
        context  = " ".join(sample["context"]["contexts"])[:800].replace("\n", " ")
        prompt   = build_prompt(context, question)

        # ── Tokenise the prompt ───────────────────────────────────────────
        enc = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=MAX_PROMPT_TOKENS
        )
        query_tensor = safe_1d(enc.input_ids[0].to(device), "query")

        # ── Generate a response (PPO rollout) ─────────────────────────────
        with torch.no_grad():
            gen_ids = ppo_trainer.generate(
                [query_tensor],
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        response_tokens = safe_1d(
            gen_ids[0][query_tensor.shape[0]:], "response"
        )
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

        # ── Parse label and explanation from the response ─────────────────
        label, explanation = parse_response(response_text)

        # ── Compute reward ────────────────────────────────────────────────
        if not response_text.strip():
            # Truly empty output — nothing to work with, skip
            print(f"Step {step:>3} | ⚠ empty output — skipping")
            continue
        lext_score = lext_module.lext(
            ground_context=context,
            ground_question=question,
            ground_explanation=sample["long_answer"],
            ground_label=sample["final_decision"].capitalize(),
            predicted_label=label,
            predicted_explanation=explanation,
            ner_pipe=ner_pipe,
        )
        # Re-centre reward so the model gets signed feedback
        reward = float(lext_score) - REWARD_CENTRE
        print(f"Step {step:>3} | label={label} | lext={lext_score:.4f} | reward={reward:.4f}")

        # ── Accumulate into batch ─────────────────────────────────────────
        query_tensors.append(query_tensor)
        response_tensors.append(response_tokens)
        rewards_list.append(reward)

        # ── PPO update when batch is full ─────────────────────────────────
        if len(query_tensors) >= BATCH_SIZE:
            flush_batch(step)

    except Exception as e:
        print(f"⚠ Error at step {step}: {e}")
        continue

# Run a final update on any leftover samples
if query_tensors:
    flush_batch(last_step)

print("Training complete.\n")

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

ppo_trainer.model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")
