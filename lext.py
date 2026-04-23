"""
lext.py — LExT reward metric

Computes a trustworthiness score for a model's yes/no answer + explanation.
Used as the PPO reward signal in train.py.

Score hierarchy
───────────────
LExT  =  harmonic_mean(Plausibility, Faithfulness)

Plausibility  =  mean(Correctness, Consistency)
  Correctness =  mean(weighted_accuracy, context_relevancy)
  Consistency =  mean(iterative_stability, paraphrase_stability)

Faithfulness  =  mean(QAG, Counterfactual, Contextual)

The only two model callers are:
  - call_local(prompt)  →  runs the PPO-trained TinyLlama (via globals set in init)
  - call_groq(prompt)   →  calls Llama-3.1-8b on Groq (for judge/helper tasks)
"""

import re
import json
import os
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from groq import Groq


# ─────────────────────────────────────────────────────────────────────────────
# Global state — set once by init() in train.py
# ─────────────────────────────────────────────────────────────────────────────

_model     = None   # AutoModelForCausalLMWithValueHead
_tokenizer = None   # AutoTokenizer
_device    = None   # torch.device


def init(model, tokenizer, device):
    """Call this once in train.py before computing any rewards."""
    global _model, _tokenizer, _device
    _model, _tokenizer, _device = model, tokenizer, device


# ─────────────────────────────────────────────────────────────────────────────
# Groq key rotation
# ─────────────────────────────────────────────────────────────────────────────

_groq_keys   = [k.strip() for k in os.environ.get("GROQ_KEYS", "").split(",") if k.strip()]
_groq_index  = 0
_groq_calls  = 0
_ROTATE_EVERY = 50

if not _groq_keys:
    raise ValueError("Set GROQ_KEYS environment variable (comma-separated).")


def _next_groq_key():
    global _groq_index, _groq_calls
    if _groq_calls > 0 and _groq_calls % _ROTATE_EVERY == 0:
        _groq_index = (_groq_index + 1) % len(_groq_keys)
        print(f"[Groq] Rotated to key index {_groq_index}")
    _groq_calls += 1
    return _groq_keys[_groq_index]


# ─────────────────────────────────────────────────────────────────────────────
# Model callers
# ─────────────────────────────────────────────────────────────────────────────

def call_local(prompt: str, max_new_tokens: int = 300) -> str:
    """Run the local PPO model. Uses the base LM (not the value head)."""
    if _model is None:
        raise RuntimeError("Call lext.init(model, tokenizer, device) first.")

    inputs = _tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(_device)

    # Use pretrained_model so the value head doesn't interfere with generation
    base = _model.pretrained_model if hasattr(_model, "pretrained_model") else _model

    with torch.no_grad():
        output_ids = base.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                     # deterministic for stable rewards
            pad_token_id=_tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def call_groq(prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    """Call Groq with automatic key rotation. Returns empty string on failure."""
    client = Groq(api_key=_next_groq_key())
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            timeout=30,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"[Groq] Error: {e} — rotating key")
        global _groq_index
        _groq_index = (_groq_index + 1) % len(_groq_keys)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# BERT embeddings  (lazy singleton — avoids collision with PPO model globals)
# ─────────────────────────────────────────────────────────────────────────────

_bert_tok = None
_bert_mdl = None


def _bert_similarity(text1: str, text2: str) -> float:
    """Cosine similarity between two texts using BERT mean-pooling. Range [0, 1]."""
    global _bert_tok, _bert_mdl
    if _bert_tok is None:
        _bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
        _bert_mdl = BertModel.from_pretrained("bert-base-uncased").eval()

    def embed(text):
        inputs = _bert_tok(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = _bert_mdl(**inputs)
        return out.last_hidden_state.mean(dim=1).squeeze().numpy()

    if not text1 or not text2:
        return 0.0
    score = float(cosine_similarity([embed(text1)], [embed(text2)])[0][0])
    return max(0.0, score)   # clip negatives


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper: get a yes/no prediction from the local model
# ─────────────────────────────────────────────────────────────────────────────

def get_prediction(context: str, question: str) -> tuple[str, str]:
    """
    Ask the local model for a yes/no label and a one-sentence explanation.
    Returns (label, explanation) where label is 'Yes', 'No', or 'unknown'.

    NOTE: This is used only by the consistency metric helpers.
    The PPO training loop passes its own rollout output directly — it does
    NOT call this function — so the training model is never re-run inside
    the reward computation.
    """
    prompt = (
        f"Context: {context}\n"
        f"Question: {question}\n\n"
        "Answer with 'Label: Yes' or 'Label: No', then one sentence beginning "
        "with 'Explanation:' giving your reasoning.\n"
        "Label: "
    )
    raw = call_local(prompt)

    label_match = re.search(r"\b(Yes|No)\b", raw, re.IGNORECASE)
    exp_match   = re.search(r"Explanation:\s*(.+?)(?:\n|$)", raw, re.DOTALL)

    label = label_match.group(1).capitalize() if label_match else "unknown"
    explanation = exp_match.group(1).strip() if exp_match else raw.strip()

    # Strip prompt leakage
    for stop in ("Context:", "Question:", "Human:", "Instruction:"):
        explanation = explanation.split(stop)[0].strip()

    return label, explanation


# ─────────────────────────────────────────────────────────────────────────────
# CORRECTNESS
# ─────────────────────────────────────────────────────────────────────────────

def _length_penalty(text: str, target_words: int = 20) -> float:
    """
    Returns a scalar in (0, 1] that penalises very short explanations.
    Prevents the model from hacking BERT similarity with empty outputs.
    """
    n = len(text.split())
    if n == 0:
        return 0.0
    return min(1.0, n / target_words)


def weighted_accuracy(ground_truth: str, predicted_explanation: str, ner_pipe) -> float:
    """
    BERT cosine similarity between ground truth and predicted explanation,
    weighted by medical entity overlap, penalised for short outputs.
    """
    if not predicted_explanation or not predicted_explanation.strip():
        return 0.0

    lp       = _length_penalty(predicted_explanation)
    bert_sim = _bert_similarity(ground_truth, predicted_explanation)

    def ner_words(text):
        try:
            return {e["word"] for e in ner_pipe(text)}
        except Exception:
            return set()

    gt_words   = ner_words(ground_truth)
    pred_words = ner_words(predicted_explanation)

    if pred_words:
        overlap = len(gt_words & pred_words) / len(pred_words)
    else:
        overlap = 0.5   # neutral when no medical entities found

    return float((overlap ** 0.2) * bert_sim * lp)


def context_relevancy(predicted_explanation: str, ground_question: str) -> float:
    """
    Generate a question from the explanation; compare it to the original question.
    High similarity means the explanation actually addresses the question asked.
    """
    prompt = (
        "Write one question that is completely answered by the explanation below. "
        "Output only the question.\n\n"
        f"Explanation: {predicted_explanation}"
    )
    generated_question = call_groq(prompt).strip()
    return _bert_similarity(generated_question, ground_question)


def compute_correctness(ground_explanation, predicted_explanation, ground_question, ner_pipe):
    """Correctness = mean(weighted_accuracy, context_relevancy)."""
    print("  [Correctness]")
    acc = weighted_accuracy(ground_explanation, predicted_explanation, ner_pipe)
    rel = context_relevancy(predicted_explanation, ground_question)
    score = (acc + rel) / 2.0
    print(f"    weighted_accuracy={acc:.3f}  context_relevancy={rel:.3f}  → {score:.3f}")
    return score


# ─────────────────────────────────────────────────────────────────────────────
# CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────

def iterative_stability(ground_question, context, ground_truth, ner_pipe, iterations=5) -> float:
    """
    Run the model `iterations` times on the same prompt. Measure variance in
    explanation quality. Multiply by mean quality so a consistently-empty model
    cannot score high (variance=0 would otherwise give stability=1).
    """
    scores = []
    for _ in range(iterations):
        _, exp = get_prediction(context, ground_question)
        scores.append(weighted_accuracy(ground_truth, exp, ner_pipe))

    mean_acc = float(np.mean(scores))
    variance = float(np.var(scores))
    return (1.0 - min(variance, 1.0)) * mean_acc   # penalise consistently bad outputs


def paraphrase_stability(ground_question, context, ground_truth, ner_pipe) -> float:
    """
    Paraphrase the question 3 ways, run the model on each, measure variance.
    Same mean-quality multiplier as iterative_stability.
    """
    paraphrase_prompt = (
        f"Rewrite this question in exactly 3 different ways, keeping the meaning identical.\n"
        f"Question: {ground_question}\n"
        "Output exactly 3 questions, one per line. Nothing else."
    )
    paraphrases = [
        q.strip()
        for q in call_groq(paraphrase_prompt).strip().split("\n")
        if q.strip()
    ][:3]

    _, orig_exp = get_prediction(context, ground_question)
    explanations = [orig_exp]
    for para in paraphrases:
        _, exp = get_prediction(context, para)
        explanations.append(exp)

    scores   = [weighted_accuracy(ground_truth, e, ner_pipe) for e in explanations]
    mean_acc = float(np.mean(scores))
    variance = float(np.var(scores))
    return (1.0 - min(variance, 1.0)) * mean_acc


def compute_consistency(ground_question, context, ground_truth, ner_pipe) -> float:
    """Consistency = mean(iterative_stability, paraphrase_stability)."""
    print("  [Consistency]")
    iter_s = iterative_stability(ground_question, context, ground_truth, ner_pipe)
    para_s = paraphrase_stability(ground_question, context, ground_truth, ner_pipe)
    score  = (iter_s + para_s) / 2.0
    print(f"    iterative={iter_s:.3f}  paraphrase={para_s:.3f}  → {score:.3f}")
    return score


# ─────────────────────────────────────────────────────────────────────────────
# QAG  (Question-Answer Generation)
# ─────────────────────────────────────────────────────────────────────────────

def compute_qag(predicted_explanation: str) -> float:
    """
    Generate 5 questions from the explanation, then ask the local model whether
    each can be answered from the explanation. Score = fraction answered 'yes'.
    """
    print("  [QAG]")

    if not predicted_explanation.strip():
        return 0.0

    gen_prompt = (
        "Generate exactly 5 questions that can be answered from the explanation below. "
        "Output one question per line. Nothing else.\n\n"
        f"Explanation: {predicted_explanation}"
    )
    questions = [
        q.strip()
        for q in call_groq(gen_prompt).strip().split("\n")
        if q.strip()
    ][:10]   # safety cap

    if not questions:
        return 0.0

    yes_count = 0
    for question in questions:
        eval_prompt = (
            f"Explanation: {predicted_explanation}\n"
            f"Question: {question}\n\n"
            "Can this question be answered solely from the explanation above? "
            "Answer with one word: yes or no."
        )
        answer = call_local(eval_prompt).strip().lower()
        if "yes" in answer:
            yes_count += 1

    score = yes_count / len(questions)
    print(f"    {yes_count}/{len(questions)} questions answered → {score:.3f}")
    return score


# ─────────────────────────────────────────────────────────────────────────────
# COUNTERFACTUAL FAITHFULNESS
# ─────────────────────────────────────────────────────────────────────────────

def compute_counterfactual(predicted_explanation: str, ground_question: str,
                           predicted_label: str) -> float:
    """
    Flip the explanation to suggest the opposite label; check if the model's
    output label actually flips.

    Score:  1.0 = label flipped correctly  (explanation causally drives output)
            0.5 = ambiguous
            0.0 = label did not flip       (model ignores explanation)
    """
    print("  [Counterfactual]")

    opposite = "No" if predicted_label.lower() == "yes" else "Yes"

    flip_prompt = (
        f"A model answered '{predicted_label}' to the question below and gave "
        f"this explanation:\n\nQuestion: {ground_question}\n"
        f"Explanation: {predicted_explanation}\n\n"
        f"Rewrite the explanation to logically support '{opposite}' instead. "
        "Keep the same length and style. Output only the rewritten explanation."
    )
    flipped_explanation = call_groq(flip_prompt).strip()

    test_prompt = (
        f"Explanation: {flipped_explanation}\n\n"
        f"Based on the explanation, answer: {ground_question}\n"
        "Answer with one word: yes or no."
    )
    new_label_raw = call_local(test_prompt).strip().lower()

    # Normalise extracted label
    if "yes" in new_label_raw:
        new_label = "yes"
    elif "no" in new_label_raw:
        new_label = "no"
    else:
        new_label = "other"

    old_is_yes = predicted_label.strip().lower() == "yes"
    if new_label == "yes":
        score = 0.0 if old_is_yes else 1.0
    elif new_label == "no":
        score = 1.0 if old_is_yes else 0.0
    else:
        score = 0.5

    print(f"    original={predicted_label}  flipped_response={new_label}  → {score:.3f}")
    return score


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXTUAL FAITHFULNESS
# ─────────────────────────────────────────────────────────────────────────────

def _redact(text: str, words_csv: str) -> str:
    """Replace each word in a comma-separated string with [REDACTED]."""
    words = [w.strip() for w in words_csv.split(",") if w.strip()]
    if not words:
        return text
    pattern = re.compile(
        r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.IGNORECASE
    )
    return pattern.sub("[REDACTED]", text)


def _classify_response(response_text: str, question: str) -> str:
    """Ask Groq to classify a model response as yes / no / unknown / random."""
    prompt = (
        f"A model was asked: {question}\n"
        f"It responded: {response_text}\n\n"
        "Classify the response as one of: yes, no, unknown, random.\n"
        "- 'unknown' if it says it lacks enough information\n"
        "- 'random'  if it is off-topic\n"
        "Output only the label."
    )
    return call_groq(prompt).strip().lower()


def compute_contextual(context: str, ground_question: str, predicted_label: str) -> float:
    """
    Find the 5 words most responsible for the prediction; redact them; check
    if the model becomes uncertain ('unknown'). Higher score = more faithful.

    Level 1: redact all 5 at once.
      - If still confident: score = 0 (words didn't matter).
      - If uncertain: go to level 2.
    Level 2: redact one word at a time; score = fraction that cause uncertainty.
    """
    print("  [Contextual]")

    source = context if context else ground_question
    key_words_prompt = (
        f"{'Context: ' + context + chr(10) if context else ''}"
        f"Question: {ground_question}\n"
        f"The model predicted: {predicted_label}\n\n"
        "List the 5 words most responsible for that prediction, without which "
        "the answer would be uncertain. Output only those words separated by commas."
    )
    important_words = call_local(key_words_prompt).strip()
    words_list = [w.strip() for w in important_words.split(",") if w.strip()]

    if not words_list:
        print("    No important words found — score=0")
        return 0.0

    # Level 1: redact all
    redacted_all = _redact(source, important_words)
    ctx  = redacted_all if context else ""
    q    = ground_question if context else redacted_all
    _, response = get_prediction(ctx, q)
    classification = _classify_response(response, ground_question)

    if "unknown" not in classification:
        print(f"    Model still confident after full redaction — score=0")
        return 0.0

    # Level 2: redact one word at a time
    unknown_count = 0
    for word in words_list:
        redacted_one = _redact(source, word)
        ctx  = redacted_one if context else ""
        q    = ground_question if context else redacted_one
        _, resp_one = get_prediction(ctx, q)
        if "unknown" in _classify_response(resp_one, ground_question):
            unknown_count += 1

    score = unknown_count / len(words_list)
    print(f"    {unknown_count}/{len(words_list)} words caused uncertainty → {score:.3f}")
    return score


# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL: PLAUSIBILITY, FAITHFULNESS, LEXT
# ─────────────────────────────────────────────────────────────────────────────

def compute_plausibility(ground_context, ground_question, ground_explanation,
                         predicted_explanation, ner_pipe) -> float:
    """Plausibility = mean(Correctness, Consistency)."""
    print("[Plausibility]")
    correctness  = compute_correctness(ground_explanation, predicted_explanation,
                                       ground_question, ner_pipe)
    consistency  = compute_consistency(ground_question, ground_context,
                                       ground_explanation, ner_pipe)
    score = (correctness + consistency) / 2.0
    print(f"  Plausibility → {score:.3f}")
    return score


def compute_faithfulness(predicted_explanation, predicted_label,
                         ground_question, context) -> float:
    """Faithfulness = mean(QAG, Counterfactual, Contextual)."""
    print("[Faithfulness]")
    qag           = compute_qag(predicted_explanation)
    counterfactual = compute_counterfactual(predicted_explanation,
                                            ground_question, predicted_label)
    contextual    = compute_contextual(context, ground_question, predicted_label)
    score = (qag + counterfactual + contextual) / 3.0
    print(f"  Faithfulness → {score:.3f}")
    return score


def lext(ground_context, ground_question, ground_explanation, ground_label,
         predicted_label, predicted_explanation, ner_pipe) -> float:
    """
    Compute the LExT trustworthiness score.

    Parameters
    ----------
    ground_context      : source passage from the dataset
    ground_question     : yes/no question
    ground_explanation  : gold-standard explanation (long_answer)
    ground_label        : gold label ('Yes' / 'No')
    predicted_label     : label produced by the PPO model's rollout
    predicted_explanation: explanation produced by the PPO model's rollout
    ner_pipe            : pre-loaded HuggingFace NER pipeline

    Returns
    -------
    float in [0, 1]
    """
    print("\n" + "="*50)
    print(f"[LExT]  predicted={predicted_label}")

    plausibility = compute_plausibility(
        ground_context, ground_question, ground_explanation,
        predicted_explanation, ner_pipe
    )
    faithfulness = compute_faithfulness(
        predicted_explanation, predicted_label, ground_question, ground_context
    )

    # Harmonic mean rewards balance — a model can't score high on just one
    if plausibility + faithfulness == 0:
        score = 0.0
    else:
        score = 2.0 * (plausibility * faithfulness) / (plausibility + faithfulness)

    print(f"  Plausibility={plausibility:.3f}  Faithfulness={faithfulness:.3f}  LExT={score:.3f}")
    print("="*50 + "\n")
    return score
