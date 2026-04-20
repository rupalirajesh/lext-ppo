import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

from metrics.trustworthiness import lext
from transformers import pipeline

import sys
sys.path.append("/content/lext-ppo")

# =========================
# Config
# =========================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SAVE_PATH = "data/tinyllama/tinyllama_pubmedqa_lext.csv"
MAX_SAMPLES = 500  # half-ish subset for Colab

# =========================
# Load Model (QLoRA style)
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def load_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        load_in_4bit=True
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
    return model

model = load_model()

# =========================
# PPO Config
# =========================
config = PPOConfig(
    learning_rate=1e-6,
    batch_size=1,
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
    aggregation_strategy='simple'
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

BATCH_SIZE = 4  # start small (Colab safe)

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

        query_tensor = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        response_tensor = ppo_trainer.generate(
            query_tensor,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7
        )

        response_text = tokenizer.decode(response_tensor[0], skip_special_tokens=True)

        # Extract parts (simple parsing)
        label = "unknown"
        explanation = response_text

        if "Answer:" in response_text:
            label = response_text.split("Answer:")[-1].split("\n")[0].strip()

        if "Reasoning:" in response_text:
            explanation = response_text.split("Reasoning:")[-1].strip()


        groq_api = ""

        # Compute LEXT reward
        reward = lext(
            context,
            question,
            sample["long_answer"],
            sample["final_decision"].capitalize(),
            model,  # no external model
            groq_api,
            ner_pipe,
            {
                "predicted_label": label,
                "predicted_explanation": explanation
            }
        )

        reward_tensor = torch.tensor([reward]).to(model.device)

        # Store rollout
        query_tensors.append(query_tensor[0])
        response_tensors.append(response_tensor[0])
        rewards.append(reward)

        print(f"Step {i} | Reward: {reward}")

        # Run PPO update every BATCH_SIZE samples
        if len(query_tensors) >= BATCH_SIZE:
            reward_tensors = torch.tensor(rewards).to(model.device)

            stats = ppo_trainer.step(
                query_tensors,
                response_tensors,
                reward_tensors
            )

            print(f"🔁 PPO update at step {i} | Mean reward: {sum(rewards)/len(rewards):.4f}")

            # Clear buffers
            query_tensors = []
            response_tensors = []
            rewards = []

    except Exception as e:
        print(f"Error at step {i}: {e}")

if len(query_tensors) > 0:
    reward_tensors = torch.tensor(rewards).to(model.device)

    ppo_trainer.step(
        query_tensors,
        response_tensors,
        reward_tensors
    )

    print("Final PPO update done.")

print("Training complete.")
