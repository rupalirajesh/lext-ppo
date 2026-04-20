import pandas as pd
from transformers import pipeline
from groq import Groq
# from langchain_ollama import OllamaLLM
import json
import re

import itertools

import os

keys = os.environ.get("GROQ_KEYS", "")
GROQ_KEYS = [k.strip() for k in keys.split(",") if k.strip()]

if len(GROQ_KEYS) == 0:
    raise ValueError("No GROQ_KEYS found. Set them via environment variables.")

class GroqKeyManager:
    def __init__(self, keys, switch_every=1000):
        self.keys = keys
        self.switch_every = switch_every
        self.call_count = 0
        self.current_index = 0

    def get_key(self):
        # 🔁 Switch key every N calls
        if self.call_count > 0 and self.call_count % self.switch_every == 0:
            self.current_index = (self.current_index + 1) % len(self.keys)
            print(f"\n🔁 Switching to Groq key {self.current_index}\n")

        self.call_count += 1
        return self.keys[self.current_index]


# 🔹 Initialize globally
key_manager = GroqKeyManager(GROQ_KEYS, switch_every=50)

def clean_output(text):
    # ✅ Normalize first (CRITICAL FIX)
    if isinstance(text, list):
        text = " ".join(str(t) for t in text)

    if not isinstance(text, str):
        text = str(text)

    # Remove code blocks
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Remove known leakage patterns
    text = re.split(r"(Human:|Instruction:|User:|Question:)", text)[0]

    # Remove repeated labels
    text = re.sub(r"(Label:.*?Explanation:)", "Label:", text, flags=re.DOTALL)

    # Normalize whitespace
    text = re.sub(r"\n+", "\n", text)

    return text.strip()

def call_model(prompt, target_model):
    llm = OllamaLLM(
        model=target_model,
        temperature=0,
        top_p=1,
        num_ctx=4096,
        num_predict=300   # increased
    )

    prediction = llm.invoke(prompt)  # ❌ NO stop tokens

    # ✅ normalize
    if isinstance(prediction, list):
        prediction = " ".join(prediction)

    return str(prediction)

def call_llama(prompt, groq_key=None, model="llama-3.1-8b-instant"):
    """
    Call the bigger Llama model using Groq with automatic key rotation.
    """
    from groq import Groq

    # 🔹 Get rotating key
    api_key = key_manager.get_key()
    
    client = Groq(api_key=api_key)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}\n\n"
                }
            ],
            model=model,
            timeout=30  # ✅ prevents hanging
        )

        result = chat_completion.choices[0].message.content
        return result

    except Exception as e:
        print(f"Groq error: {e}")

        # 🔁 OPTIONAL: switch key immediately on failure
        key_manager.current_index = (key_manager.current_index + 1) % len(key_manager.keys)
        print("🔁 Switched key due to error")

        return ""

def get_prediction(context, question, target_model, groq_key=None, include_context=True, verbose=False):
    """
    Get the prediction from the target model.
    The prompt is constructed using the context and question.
    """
    prompt = f"""
    Label: Yes or No
    Explanation: One short sentence.
    
    Q: {question}
    C: {context if include_context and context else ""}
    """
    
    prediction_raw = call_model(prompt, target_model)
    prediction_raw = prediction_raw[:1000]

    if verbose:
        print("\n===== RAW MODEL OUTPUT =====")
        print(prediction_raw)
        print("============================\n")
    
    # 🔹 Extraction prompt (fixed)
    extraction_prompt = (
        f"Prediction: {prediction_raw}\n"
        "Extract this into two parts: 'label' and 'explanation'. "
        "Return ONLY a valid JSON object with keys 'label' and 'explanation'. "
        "Do NOT include markdown, code fences, or extra text."
    )
    
    label_match = re.search(r"Label:\s*(Yes|No)", prediction_raw, re.IGNORECASE)
    explanation_match = re.search(
        r"Explanation:\s*(.*?)(?:\nLabel:|\n[A-Z][a-z]+:|$)",
        prediction_raw,
        re.DOTALL
    )

    if label_match and explanation_match:
        label = label_match.group(1).capitalize()
        explanation = explanation_match.group(1).strip()
        explanation = explanation.split("Human:")[0]
        explanation = explanation.split("Instruction:")[0]
        explanation = explanation.split("Question:")[0]
        explanation = explanation.strip()
        return label, explanation
    
    extraction = call_llama(extraction_prompt, groq_key)
    extraction = clean_output(extraction)
    # 🔹 Clean markdown/code fences
    cleaned = re.sub(r"```json|```", "", extraction).strip()
    
    # 🔹 Extract JSON block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)

    try:
        if match:
            json_str = match.group(0)
            result = json.loads(json_str)

            label = result.get("label", "unknown")
            explanation = result.get("explanation", "")
            explanation = explanation.split("Human:")[0]
            explanation = explanation.split("Instruction:")[0]
            explanation = explanation.split("Question:")[0]
            explanation = explanation.strip()

            # 🔹 Normalize label
            if isinstance(label, list):
                label = " ".join(label)
                label = str(label).lower()

            if "yes" in label:
                label = "Yes"
            elif "no" in label:
                label = "No"
            else:
                label = "unknown"

        else:
            raise ValueError("No JSON found")

    except Exception:
        text = cleaned.lower()

        if "yes" in text:
            label = "Yes"
        elif "no" in text:
            label = "No"
        else:
            label = "unknown"

        explanation = cleaned
    
    return label, explanation
