import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

from transformers import pipeline
import sys
sys.path.append("/content/lext-ppo")

from metrics.trustworthiness import lext

# =========================
# Config
# =========================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_SAMPLES = 500
BATCH_SIZE = 4

# =========================
# Load Model
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # important

def load_model():
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
    return model

model = load_model()

# =========================
# PPO Config
# =========================
config = PPOConfig(
    learning_rate=1e-6,
    batch_size=BATCH_SIZE,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    ppo_epochs=1,
    log_with=None
)

ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    tokenizer=tokenizer
)

device = ppo_trainer.accelerator.device
model.to(device)

# =========================
# Dataset
# =========================
dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
dataset = dataset.select(range(MAX_SAMPLES))

# =========================
# NER (for LEXT)
# =========================
ner_pipe = pipeline(
    "token-classification",
    model="Clinical-AI-Apollo/Medical-NER",
    aggregation_strategy='simple',
    device=0 if torch.cuda.is_available() else -1
)

# =========================
# Prompt builder
# =========================
def build_prompt(context, question):
    return f"""
You are a medical assistant.

Context: {context}

Question: {question}

Respond EXACTLY in this format:

Answer: Yes or No
Reasoning: step-by-step reasoning
Confidence: low/medium/high
"""

# =========================
# Training Loop
# =========================
print("Starting PPO training...")

query_tensors = []
response_tensors = []
rewards = []

for i, sample in enumerate(dataset):
    try:
        if sample["final_decision"].lower() == "maybe":
            continue

        question = sample["question"]
        context = " ".join(sample["context"]["contexts"])
        context = context[:800].replace("\n", " ")

        prompt = build_prompt(context, question)

        # =========================
        # Tokenize
        # =========================
        query_tensor = tokenizer(prompt, return_tensors="pt").input_ids[0].to(device)
        query_tensor = query_tensor.squeeze()

        # =========================
        # Generate
        # =========================
        response_tensor = ppo_trainer.generate(
            query_tensor.unsqueeze(0),
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7
        )[0]

        # =========================
        # Extract generated tokens ONLY
        # =========================
        generated_tokens = response_tensor[len(query_tensor):]
        generated_tokens = generated_tokens.squeeze()

        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # =========================
        # Parse output
        # =========================
        label = "unknown"
        explanation = response_text

        if "Answer:" in response_text:
            label = response_text.split("Answer:")[-1].split("\n")[0].strip()

        if "Reasoning:" in response_text:
            explanation = response_text.split("Reasoning:")[-1].strip()

        # =========================
        # Reward
        # =========================
        reward = lext(
            context,
            question,
            sample["long_answer"],
            sample["final_decision"].capitalize(),
            model,
            "",
            ner_pipe,
            {
                "predicted_label": label,
                "predicted_explanation": explanation
            }
        )

        # =========================
        # Store rollout
        # =========================
        query_tensors.append(query_tensor)
        response_tensors.append(generated_tokens)  # FIXED
        rewards.append(float(reward))

        print(f"Step {i} | Reward: {reward:.4f}")

        # =========================
        # PPO step
        # =========================
        if len(query_tensors) >= BATCH_SIZE:
            reward_tensors = torch.tensor(rewards).to(device)

            ppo_trainer.step(
                query_tensors,
                response_tensors,
                reward_tensors
            )

            print(f"🔁 PPO update at step {i}")

            query_tensors, response_tensors, rewards = [], [], []

    except Exception as e:
        print(f"Error at step {i}: {e}")

# =========================
# Final step
# =========================
if len(query_tensors) > 0:
    reward_tensors = torch.tensor(rewards).to(device)
    ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

print("Training complete.")
