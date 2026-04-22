import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import pipeline
import sys

sys.path.append("/content/lext-ppo")
sys.path.append("/content/lext-ppo/src")

import basic_functions

# =========================
# Config
# =========================
MODEL_NAME        = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_SAMPLES       = 500
BATCH_SIZE        = 4
MAX_PROMPT_TOKENS = 512
MAX_NEW_TOKENS    = 150

# =========================
# Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# =========================
# Model
# =========================
model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)

# =========================
# PPO Config + Trainer
# =========================
ppo_config = PPOConfig(
    learning_rate=1e-6,
    batch_size=BATCH_SIZE,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    ppo_epochs=1,
    log_with=None,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
)

device = ppo_trainer.accelerator.device
model.to(device)

basic_functions.init_globals(model, tokenizer, device)

# =========================
# Dataset
# =========================
dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
dataset = dataset.select(range(MAX_SAMPLES))

# =========================
# NER pipeline (for LEXT)
# =========================
ner_pipe = pipeline(
    "token-classification",
    model="Clinical-AI-Apollo/Medical-NER",
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1,
)

# =========================
# Helpers
# =========================
def build_prompt(context: str, question: str) -> str:
    return (
        "You are a medical assistant.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Respond EXACTLY in this format:\n\n"
        "Answer: Yes or No\n"
        "Reasoning: step-by-step reasoning\n"
        "Confidence: low/medium/high\n"
    )


def safe_1d(t: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """Guarantee a 1-D cpu-safe tensor; raise early with a clear message."""
    t = t.detach().flatten()
    if t.dim() != 1 or t.numel() == 0:
        raise ValueError(f"{name} has bad shape after flatten: {t.shape}")
    return t


def parse_response(text: str):
    label       = "unknown"
    explanation = text

    if "Answer:" in text:
        raw = text.split("Answer:")[-1].split("\n")[0].strip().lower()
        if "yes" in raw:
            label = "Yes"
        elif "no" in raw:
            label = "No"

    if "Reasoning:" in text:
        explanation = text.split("Reasoning:")[-1].strip()

    return label, explanation


# =========================
# Rollout buffers
# =========================
query_tensors    = []
response_tensors = []
rewards_list     = []


def flush_ppo_step(step_idx: int):
    reward_tensors = [
        torch.tensor(r, dtype=torch.float32)
        for r in rewards_list
    ]
    ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
    print(f"  🔁 PPO update at step {step_idx} "
          f"(mean reward {sum(rewards_list)/len(rewards_list):.4f})")
    query_tensors.clear()
    response_tensors.clear()
    rewards_list.clear()

from metrics.trustworthiness import lext

# =========================
# Training loop
# =========================
print("Starting PPO training…")

for i, sample in enumerate(dataset):
    try:
        decision = sample["final_decision"].strip().lower()
        if decision == "maybe":
            continue

        question = sample["question"]
        context  = " ".join(sample["context"]["contexts"])
        context  = context[:800].replace("\n", " ")
        prompt   = build_prompt(context, question)

        # ── Tokenise ──────────────────────────────────────────────────────
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_PROMPT_TOKENS,
        )
        # safe_1d gives us a guaranteed 1-D tensor
        query_tensor = safe_1d(enc.input_ids[0].to(device), "query_tensor")

        # ── Generate ──────────────────────────────────────────────────────
        # TRL 0.7.x requires a LIST of 1-D tensors, NOT a batched 2-D tensor
        with torch.no_grad():
            gen_ids = ppo_trainer.generate(
                [query_tensor],                      # ← key fix: list of 1-D
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_ids         = safe_1d(gen_ids[0], "full_ids")
        generated_tokens = safe_1d(
            full_ids[query_tensor.shape[0]:], "generated_tokens"
        )

        response_text = tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        # ── Parse ─────────────────────────────────────────────────────────
        label, explanation = parse_response(response_text)

        # ── Reward ────────────────────────────────────────────────────────
        reward = float(lext(
            context,
            question,
            sample["long_answer"],
            sample["final_decision"].capitalize(),
            model,
            "",
            ner_pipe,
            {
                "predicted_label":       label,
                "predicted_explanation": explanation,
            },
        ))

        # ── Accumulate ────────────────────────────────────────────────────
        query_tensors.append(query_tensor)
        response_tensors.append(generated_tokens)
        rewards_list.append(reward)

        print(f"Step {i:>3} | label={label:<8} | reward={reward:.4f}")

        # ── PPO step ──────────────────────────────────────────────────────
        if len(query_tensors) >= BATCH_SIZE:
            flush_ppo_step(i)

    except Exception as e:
        print(f"⚠️  Error at step {i}: {e}")
        continue

# Final partial batch
if query_tensors:
    flush_ppo_step(i)

print("✅ Training complete.")

# Save the final trained model
save_path = "/content/drive/MyDrive/tinyllama_ppo_finetuned"

ppo_trainer.model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ Model saved to {save_path}")
