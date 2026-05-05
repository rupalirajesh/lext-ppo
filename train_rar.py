"""
train_rar.py — PPO fine-tuning of TinyLlama using LExT-RaR as the reward

Run with:
    GROQ_KEYS=key1,key2 python train_rar.py

Differences from the original train.py + lext.py
──────────────────────────────────────────────────
- No BERT, no NER pipeline. All 7 submetrics scored in ONE Groq call.
- reward function is lext_rar() defined right here in this file.
- Model saved to a different path (tinyllama_ppo_lext_rar) so it never
  collides with the original lext run.
- Scores logged to /content/drive/MyDrive/tinyllama_ppo_lext_rar/scores.csv
  with columns: step, plausibility, faithfulness, lext_score, reward

Everything else (PPO config, dataset, prompting, batch logic) is identical
to the original so results are directly comparable.
"""

import os
import re
import csv
import torch
from groq import Groq
from transformers import AutoTokenizer
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ← only things you might want to change
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME        = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SAVE_PATH         = "/content/drive/MyDrive/tinyllama_ppo_lext_rar"          # different from original
CSV_PATH          = "/content/drive/MyDrive/tinyllama_ppo_lext_rar/scores.csv"
MAX_SAMPLES       = 500
BATCH_SIZE        = 4
MAX_PROMPT_TOKENS = 384
MAX_NEW_TOKENS    = 150
LEARNING_RATE     = 1e-6
KL_COEF           = 0.2   # prevents reward hacking / mode collapse
REWARD_CENTRE     = 0.5   # shifts [0,1] reward to [-0.5, +0.5]


# ─────────────────────────────────────────────────────────────────────────────
# GROQ  — key rotation every 50 calls
# ─────────────────────────────────────────────────────────────────────────────

_groq_keys    = [k.strip() for k in os.environ.get("GROQ_KEYS", "").split(",") if k.strip()]
_groq_index   = 0
_groq_calls   = 0
_ROTATE_EVERY = 10 # changed from 50 cuz the model is now 70b

if not _groq_keys:
    raise ValueError("Set GROQ_KEYS environment variable (comma-separated API keys).")


def _next_groq_key() -> str:
    global _groq_index, _groq_calls
    if _groq_calls > 0 and _groq_calls % _ROTATE_EVERY == 0:
        _groq_index = (_groq_index + 1) % len(_groq_keys)
        print(f"[Groq] Rotated to key index {_groq_index}")
    _groq_calls += 1
    return _groq_keys[_groq_index]


import time

def call_groq(prompt: str, retries: int = 3) -> str:
    global _idx, _calls
    if _calls > 0 and _calls % _ROTATE_EVERY == 0:
        _idx = (_idx + 1) % len(_keys)
    _calls += 1
    for attempt in range(retries):
        try:
            resp = Groq(api_key=_keys[_idx]).chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                timeout=30,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 60 * (attempt + 1)   # 60s, 120s, 180s
                print(f"[Groq] Rate limit hit — waiting {wait}s before retry {attempt+1}/{retries}")
                time.sleep(wait)
                _idx = (_idx + 1) % len(_keys)   # also rotate key while waiting
            else:
                print(f"[Groq] Error: {e}")
                return ""
    print("[Groq] All retries exhausted — returning empty")
    return ""

# ─────────────────────────────────────────────────────────────────────────────
# LEXT-RAR REWARD  — single Groq call scores all 7 submetrics at once
# ─────────────────────────────────────────────────────────────────────────────

def lext_rar(
    ground_context: str,
    ground_question: str,
    ground_explanation: str,
    predicted_label: str,
    predicted_explanation: str,
) -> tuple[float, float, float]:
    """
    Score a model response with a single Groq rubric call.

    Submetrics and how they aggregate:
        Plausibility = mean(Correctness, Consistency)
            Correctness = mean(weighted_accuracy, context_relevancy)
            Consistency = mean(iterative_stability, paraphrase_stability)
        Faithfulness = mean(qag, counterfactual, contextual)
        LExT-RaR     = harmonic_mean(Plausibility, Faithfulness)

    Returns
    -------
    (lext_score, plausibility, faithfulness)  — all floats in [0, 1]
    """

    prompt = f"""You are a medical QA evaluation expert. A model was asked a yes/no medical question and gave an answer and explanation. Score the quality of the response using the 7 criteria below.

QUESTION:
{ground_question}

GROUND TRUTH EXPLANATION:
{ground_explanation}

MODEL ANSWER: {predicted_label}
MODEL EXPLANATION:
{predicted_explanation}

CONTEXT:
{ground_context}

---
Score each criterion from 0.0 to 1.0. Think carefully, then output ONLY the 7 lines below, nothing else.

1. weighted_accuracy    — How conceptually close is the model explanation to the ground truth? Same medical concepts, reasoning, and key entities?
2. context_relevancy    — Does the model explanation actually answer the question? Is it relevant and sufficient to justify the yes/no answer?
3. iterative_stability  — Does the explanation seem like it reflects consistent model behaviour, or does it look erratic / unusually lucky?
4. paraphrase_stability — If the question were worded differently, would this explanation still be a reasonable answer? Is it robust to rephrasing?
5. qag                  — Does the explanation contain enough concrete information that factual questions derived from it could be answered from it alone?
6. counterfactual       — Does the explanation causally justify the answer '{predicted_label}'? If the reasoning were flipped, would the opposite answer follow?
7. contextual           — Is the prediction tightly tied to specific entities in the context? Would removing key medical terms make the model uncertain?

weighted_accuracy: <score>
context_relevancy: <score>
iterative_stability: <score>
paraphrase_stability: <score>
qag: <score>
counterfactual: <score>
contextual: <score>"""

    raw = call_groq(prompt)

    # Parse each score — default 0.0 if missing
    def parse(key: str) -> float:
        m = re.search(rf"{key}:\s*([0-9]*\.?[0-9]+)", raw, re.IGNORECASE)
        if m:
            return max(0.0, min(1.0, float(m.group(1))))
        print(f"  [lext_rar] could not parse '{key}' — defaulting to 0.0")
        return 0.0

    wa   = parse("weighted_accuracy")
    cr   = parse("context_relevancy")
    is_  = parse("iterative_stability")
    ps   = parse("paraphrase_stability")
    qag  = parse("qag")
    cf   = parse("counterfactual")
    ctx  = parse("contextual")

    # Aggregate
    correctness  = (wa + cr)        / 2.0
    consistency  = (is_ + ps)       / 2.0
    plausibility = (correctness + consistency) / 2.0
    faithfulness = (qag + cf + ctx) / 3.0

    if plausibility + faithfulness == 0:
        lext_score = 0.0
    else:
        lext_score = 2.0 * (plausibility * faithfulness) / (plausibility + faithfulness)

    print(
        f"  [lext_rar] wa={wa:.2f} cr={cr:.2f} is={is_:.2f} ps={ps:.2f} "
        f"qag={qag:.2f} cf={cf:.2f} ctx={ctx:.2f} | "
        f"P={plausibility:.3f}  F={faithfulness:.3f}  LExT={lext_score:.3f}"
    )

    return lext_score, plausibility, faithfulness


# ─────────────────────────────────────────────────────────────────────────────
# CSV LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def init_csv(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "plausibility", "faithfulness", "lext_score", "reward"])


def log_csv(path: str, step: int, plausibility: float, faithfulness: float,
            lext_score: float, reward: float):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            step,
            f"{plausibility:.4f}",
            f"{faithfulness:.4f}",
            f"{lext_score:.4f}",
            f"{reward:.4f}",
        ])


# ─────────────────────────────────────────────────────────────────────────────
# MODEL + PPO SETUP
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
    init_kl_coef=KL_COEF,
    target=6.0,
    horizon=10_000,
    log_with=None,
)

ppo_trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer)
device = ppo_trainer.accelerator.device
model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
dataset = dataset.select(range(MAX_SAMPLES))


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(context: str, question: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system",  "content": "You are a medical assistant. Answer yes/no questions about medical research."},
            {"role": "user",    "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer with:\nAnswer: Yes or No\nReasoning: one sentence"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_response(text: str) -> tuple[str, str]:
    """Extract (label, explanation) from the model's raw output."""
    label, explanation = "unknown", text.strip()

    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("answer:"):
            raw = line.split(":", 1)[-1].strip().lower()
            if "yes" in raw:   label = "Yes"
            elif "no" in raw:  label = "No"
        elif line.lower().startswith("reasoning:"):
            explanation = line.split(":", 1)[-1].strip()

    # Fallback label scan
    if label == "unknown":
        if "yes" in text.lower(): label = "Yes"
        elif "no" in text.lower(): label = "No"

    # Fallback explanation
    if not explanation.strip():
        explanation = text.strip()

    # Strip prompt leakage
    for stop in ("Context:", "Question:", "Answer:", "<|"):
        explanation = explanation.split(stop)[0].strip()

    return label, explanation


def safe_1d(tensor: torch.Tensor, name: str = "") -> torch.Tensor:
    """Flatten to 1-D — TRL requires 1-D tensors in its step() buffers."""
    t = tensor.detach().flatten()
    if t.dim() != 1 or t.numel() == 0:
        raise ValueError(f"Tensor '{name}' has bad shape: {t.shape}")
    return t


# ─────────────────────────────────────────────────────────────────────────────
# PPO BATCH BUFFERS + FLUSH
# ─────────────────────────────────────────────────────────────────────────────

query_tensors    = []
response_tensors = []
rewards_list     = []


def flush_batch(step: int):
    """Run one PPO update then clear the buffers (always clears, even on error)."""
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
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

init_csv(CSV_PATH)
print("Starting PPO training with LExT-RaR rewards…\n")
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

        # ── Tokenise prompt ───────────────────────────────────────────────
        enc = tokenizer(prompt, return_tensors="pt",
                        truncation=True, max_length=MAX_PROMPT_TOKENS)
        query_tensor = safe_1d(enc.input_ids[0].to(device), "query")

        # ── Generate response (PPO rollout) ───────────────────────────────
        with torch.no_grad():
            gen_ids = ppo_trainer.generate(
                [query_tensor],
                max_new_tokens=MAX_NEW_TOKENS,
                return_prompt=False,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        response_tokens = safe_1d(gen_ids[0], "response")
        response_text   = tokenizer.decode(response_tokens, skip_special_tokens=True)

        # ── Parse label + explanation ─────────────────────────────────────
        label, explanation = parse_response(response_text)
        print(f"Step {step:>3} | label={label} | {explanation[:120]}")

        if not response_text.strip():
            print(f"Step {step:>3} | ⚠ empty output — skipping")
            continue

        # ── Score with LExT-RaR (one Groq call) ──────────────────────────
        lext_score, plausibility, faithfulness = lext_rar(
            ground_context=context,
            ground_question=question,
            ground_explanation=sample["long_answer"],
            predicted_label=label,
            predicted_explanation=explanation,
        )

        reward = float(lext_score) - REWARD_CENTRE
        print(f"Step {step:>3} | lext={lext_score:.4f} | reward={reward:+.4f}")

        # ── Log to CSV ────────────────────────────────────────────────────
        log_csv(CSV_PATH, step, plausibility, faithfulness, lext_score, reward)

        # ── Accumulate into batch ─────────────────────────────────────────
        query_tensors.append(query_tensor)
        response_tensors.append(response_tokens)
        rewards_list.append(reward)

        if len(query_tensors) >= BATCH_SIZE:
            flush_batch(step)

    except Exception as e:
        print(f"⚠ Error at step {step}: {e}")
        continue

# Final update for any leftover samples
if query_tensors:
    flush_batch(last_step)

print("Training complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_PATH, exist_ok=True)
ppo_trainer.model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"Model  → {SAVE_PATH}")
print(f"Scores → {CSV_PATH}")
