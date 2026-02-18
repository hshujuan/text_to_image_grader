# Soft-TIFA vs DSG: A Deep Dive into Two Leading Text-to-Image Evaluation Metrics

*A code-level comparison of GenEval2's Soft-TIFA and Google's Davidsonian Scene Graph — what they measure, how they score, and which one you should use.*

---

As text-to-image (T2I) models like DALL·E 3, Stable Diffusion, and FLUX continue to improve, a critical question remains: **how do we reliably measure whether the generated image actually matches the text prompt?**

Human evaluation is the gold standard, but it's expensive and doesn't scale. Automated metrics like FID measure image quality but ignore the prompt entirely. CLIPScore gives a holistic similarity score but misses compositional details — it can't tell you *which parts* of the prompt were satisfied and which were missed.

Enter **QG/A metrics** — Question Generation and Answering. The idea is elegant: decompose a text prompt into atomic questions, then ask a vision-language model (VLM) to verify each one against the generated image. Two major frameworks implement this paradigm:

- **Soft-TIFA**, from Meta's GenEval2 (2024)
- **DSG** (Davidsonian Scene Graph), from Google (ICLR 2024)

Both claim to evaluate "faithfulness" of generated images to text prompts. But having built a production T2I grading system using both, I've found they make fundamentally different engineering trade-offs. This article is a code-level breakdown of those differences.

---

## The Core Idea: Both Are QG/A Metrics

Both Soft-TIFA and DSG follow the same high-level pattern:

```
Text Prompt → Generate Questions → Ask VLM about Image → Aggregate Scores
```

But that's where the similarity ends. They differ in **every stage** of this pipeline.

---

## Stage 1: Question Generation

### Soft-TIFA: One Shot, Flat List

Soft-TIFA keeps question generation simple. Given a prompt like *"two red cats sitting next to a blue vase"*, it generates a flat list of question-answer pairs:

```
Q: "How many cats are in the image?"       A: "two"
Q: "Are the cats red?"                      A: "Yes"
Q: "Are there any vases in the image?"      A: "Yes"
Q: "Is the vase blue?"                      A: "Yes"
Q: "Are the cats sitting?"                  A: "Yes"
```

In GenEval2's implementation, these pairs come from one of three sources (in priority order):

1. **Pre-defined benchmark pairs** — 800 prompts in `geneval2_data.jsonl` with human-verified Q/A pairs. Free, instant, and exact.
2. **Single LLM call** — For custom prompts, one zero-shot GPT-4o call (~150 tokens) generates the Q/A list.
3. **Rule-based fallback** — A parser using dictionaries of colors, materials, spatial relations, and action verbs. Free and offline.

The result is a flat `[["question", "answer"], ...]` list with no structural relationships between questions.

### DSG: Three-Step Structured Pipeline

DSG takes a fundamentally different approach, inspired by Davidsonian formal semantics. It decomposes question generation into three sequential LLM calls, each with **23 in-context examples** drawn from the TIFA-160 benchmark:

**Step 1 — Tuple Generation:** Decompose the prompt into typed semantic tuples.

```
Input:  "two red cats sitting next to a blue vase"
Output:
  1 | entity (cat)
  2 | attribute - count (cat, two)  
  3 | attribute - color (cat, red)
  4 | attribute - state (cat, sitting)
  5 | entity (vase)
  6 | attribute - color (vase, blue)
  7 | relation - spatial (cat, vase, next to)
```

**Step 2 — Question Generation:** Convert each tuple into a natural language question.

```
  1 | Is there a cat in the image?
  2 | How many cats are in the image?
  3 | Are the cats red?
  4 | Are the cats sitting?
  5 | Is there a vase in the image?
  6 | Is the vase blue?
  7 | Are the cats next to the vase?
```

**Step 3 — Dependency Generation:** Build a dependency graph between questions.

```
  1 | 0          (root — no dependency)
  2 | 1          (count depends on cat existing)
  3 | 1          (color depends on cat existing)
  4 | 1          (state depends on cat existing)
  5 | 0          (root)
  6 | 5          (vase color depends on vase existing)
  7 | 1, 5       (spatial relation depends on BOTH existing)
```

Each step feeds 23 carefully annotated examples to the LLM as few-shot demonstrations, resulting in higher-quality, atomic, and unique questions — but at the cost of three separate LLM calls with large context windows.

### The Trade-Off

| | **Soft-TIFA** | **DSG** |
|---|---|---|
| LLM calls for QG | **1** (~150 tokens) | **3** (~3,000+ tokens with ICL examples) |
| Question quality | Good (zero-shot) | **Better** (23 few-shot examples per step) |
| Uniqueness guarantee | ❌ May generate redundant questions | ✅ Tuple decomposition ensures atomicity |
| Structural awareness | ❌ Flat list | ✅ Typed tuples + dependency graph |
| Cost per prompt | ~$0.005 | ~$0.03 (**6× more**) |

For a production system processing thousands of evaluations, this 6× cost difference adds up fast.

---

## Stage 2: VQA Scoring — The Critical Divergence

This is where the two frameworks diverge most significantly, and where Soft-TIFA holds its strongest advantage.

### Soft-TIFA: Soft Scores via Token Log-Probabilities

The "Soft" in Soft-TIFA is the key innovation. Instead of asking the VLM for a text answer and checking if it matches, Soft-TIFA examines the **raw token probabilities** from the model's output distribution.

From GenEval2's `evaluation.py`:

```python
def soft_tifa(vqa_list, image_filepath):
    score_list = []
    for question, answer in vqa_list:
        if question.startswith("How many"):
            answer_list = [answer, answer.capitalize(), 
                           ' ' + answer, ' ' + answer.capitalize(),
                           return_numeric_string(answer), 
                           ' ' + return_numeric_string(answer)]
        else:
            answer_list = ['Yes', 'yes', ' yes', ' Yes']

        pred, ans_prob = send_message_with_image(
            f'{question} Answer in one word.', image_filepath,
            answer_list=answer_list
        )
        score_list.append(ans_prob)  # ← SOFT: uses probability, not text
    return sum(score_list) / len(score_list), score_list
```

The `send_message_with_image` function extracts the softmax probabilities for all answer-variant tokens and **sums them**:

```python
def send_message_with_image(prompt, image_filepath, answer_list=None):
    outputs = model.generate(**inputs,
        max_new_tokens=1,
        output_scores=True,           # ← Request raw logits
        return_dict_in_generate=True
    )
    scores = outputs.scores[0]
    probs = torch.nn.functional.softmax(scores, dim=-1)

    lm_prob = 0
    for answer in answer_list:
        ans_token_id = tokenizer.encode(answer)[0]
        lm_prob += probs[0, ans_token_id].item()  # ← Sum token probabilities

    return pred, lm_prob
```

This produces a **continuous score between 0.0 and 1.0** for each question. A score of 0.73 means the model is "73% confident" the image contains the expected element. This captures nuance that binary scoring completely misses.

### DSG: Hard Binary Scores

DSG takes the simpler approach — it generates a text answer from the VQA model and checks for exact string match:

```python
def calc_vqa_score(qid2answer, qid2dependency=None, qid2gtanswer=None):
    qid2scores = {}
    for qid, answer in qid2answer.items():
        gt_answer = qid2gtanswer[qid]
        qid2scores[qid] = float(answer == gt_answer)  # ← HARD: 0.0 or 1.0
    ...
```

Every question gets a **0 or 1**. The model either said "yes" or it didn't. There's no middle ground — a 99% confident "yes" and a barely-above-threshold "yes" both score 1.0.

### Why This Matters

Consider evaluating the image for the prompt *"a slightly faded red car."* When asked *"Is the car red?"*:

- The VLM might output "Yes" with 65% probability (it's faded, so there's uncertainty).
- **Soft-TIFA** captures this: score = 0.65. The fading is reflected in the confidence.
- **DSG** loses this entirely: score = 1.0 (it said "yes") or 0.0 (it said "no"). The 65% vs 99% distinction vanishes.

The original GenEval2 paper specifically introduced Soft-TIFA as an improvement over hard TIFA for this reason — it provides a more discriminative signal, especially when comparing high-quality models that get most things "mostly right."

---

## Stage 3: Aggregation — Where DSG Fights Back

### DSG's Dependency Filtering

DSG's dependency graph is its strongest feature, and it directly addresses a real problem with flat evaluation.

Consider: *"a red motorcycle parked by paint-chipped doors."*

```
Q1: Is there a motorcycle? → answered "no"
Q2: Is the motorcycle red? → answered "yes" (hallucination from VLM)  
Q3: Is the motorcycle parked? → answered "yes" (hallucination from VLM)
```

With Soft-TIFA's flat averaging, Q2 and Q3 give false credit — the VLM is answering about a motorcycle that doesn't exist. The final score is inflated.

DSG's dependency tree catches this:

```python
for qid, parent_ids in qid2dependency.items():
    any_parent_answered_no = False
    for parent_id in parent_ids:
        if parent_id == 0:
            continue
        if qid2scores[parent_id] == 0:
            any_parent_answered_no = True
            break
    if any_parent_answered_no:
        qid2scores_after_filtering[qid] = 0.0  # ← Zero out child
        qid2validity[qid] = False
```

Since Q1 (parent) answered "no," Q2 and Q3 (children) are automatically **zeroed out**. You can't evaluate the color of something that doesn't exist.

This produces two scores:
- `average_score_without_dependency`: Flat average (like Soft-TIFA)
- `average_score_with_dependency`: Filtered average (DSG's contribution)

### Soft-TIFA's GM Aggregation

Soft-TIFA doesn't have dependency filtering, but it does offer **Geometric Mean (GM)** aggregation as a partial solution:

```python
if method == 'soft_tifa_gm':
    per_prompt_scores = [gmean(s) for s in all_score_lists]
```

GM is naturally harsh on outliers. If one question scores 0.0, the entire prompt's GM score is dragged down dramatically. This gives Soft-TIFA GM some of the "all-or-nothing" behavior that dependency filtering provides — though less precisely targeted.

| Aggregation | Formula | Behavior |
|---|---|---|
| Soft-TIFA AM | $(s_1 + s_2 + ... + s_n) / n$ | Forgiving — one bad score barely affects total |
| Soft-TIFA GM | $(s_1 \times s_2 \times ... \times s_n)^{1/n}$ | Harsh — one zero tanks everything |
| DSG (no dep) | Same as AM but binary | Flat average of 0s and 1s |
| DSG (with dep) | AM after zeroing invalid children | Structurally aware filtering |

---

## The VLM Bias Problem

A recent ICCV 2025 paper (Deng et al., "Leveraging Panoptic Scene Graph for Evaluating Fine-Grained Text-to-Image Generation") tested both TIFA and DSG's yes/no QA pairs against human judgments. The results were sobering:

| Metric | Human Agreement on Yes/No Pairs |
|---|---|
| Random guessing | 50% |
| **TIFA** | **64%** |
| **DSG** | **67%** |

Both are barely above coin-flip accuracy. The paper found that **VLMs tend to answer "yes" regardless of the actual image content**, a systemic bias that undermines the entire QG/A paradigm.

Soft-TIFA's logprob approach partially mitigates this — even when the model's top token is "yes," the probability might be 0.55 vs 0.95, and Soft-TIFA captures that difference. DSG's hard scoring treats both as 1.0.

---

## Practical Comparison: Running Both

Here's what it looks like in practice for the prompt *"three blue butterflies flying over a golden sunflower field"*:

### Soft-TIFA Output

```
VQA pairs (5 questions):
  "How many butterflies are in the image?" → "three"     score: 0.42
  "Are the butterflies blue?"             → "Yes"        score: 0.88
  "Are there any sunflowers in the image?" → "Yes"       score: 0.95
  "Is the sunflower field golden?"         → "Yes"        score: 0.71
  "Are the butterflies flying?"            → "Yes"        score: 0.83

Soft-TIFA AM: 75.8
Soft-TIFA GM: 72.1
```

Counting ("three") got a low score (0.42) because the model was uncertain — maybe it generated two or four. The GM is pulled down by this weak atom, signaling that counting was a problem.

### DSG Output

```
Tuples:
  1 | entity (butterfly)
  2 | attribute - count (butterfly, three)
  3 | attribute - color (butterfly, blue)
  4 | attribute - state (butterfly, flying)
  5 | entity (sunflower field)
  6 | attribute - color (sunflower field, golden)
  7 | relation - spatial (butterfly, sunflower field, over)

Dependencies:
  1 | 0      2 | 1      3 | 1      4 | 1
  5 | 0      6 | 5      7 | 1, 5

Scores:
  Q1: 1  Q2: 0  Q3: 1  Q4: 1  Q5: 1  Q6: 1  Q7: 1

Score without dependency: 85.7  (6/7)
Score with dependency:    85.7  (Q2=0, but children aren't zeroed because Q1=1)
```

DSG correctly says counting failed (Q2=0), but everything else is a hard 1. It can't distinguish between "confidently blue" (0.88) and "barely golden" (0.71) — both get 1.0.

---

## Cost Analysis

For a production system evaluating 1,000 prompts:

| Component | Soft-TIFA | DSG |
|---|---|---|
| Question generation | 1,000 LLM calls × ~150 tokens = **150K tokens** | 3,000 LLM calls × ~1,000 tokens = **3M tokens** |
| VQA scoring | 5,000 VLM calls (same for both) | 5,000 VLM calls (same for both) |
| **Total QG cost** | **~$0.15** | **~$3.00** |
| Pre-defined prompts available | 800 (free) | 0 |

If your benchmark includes GenEval2's 800 prompts, Soft-TIFA's question generation cost for those is exactly **$0**.

---

## When to Use Which

### Choose Soft-TIFA when:

- **Budget matters** — 6× cheaper per prompt for question generation
- **You need nuanced scoring** — Soft logprobs distinguish "mostly right" from "confidently right"
- **You're benchmarking against GenEval2** — 800 pre-defined prompts available for free
- **Real-time/interactive use** — Faster single-call QG
- **Comparing high-quality models** — Soft scores differentiate models that both "pass" but with different confidence

### Choose DSG when:

- **Complex multi-object scenes** — Dependency graph prevents cascading false penalties
- **Question quality is paramount** — 23 ICL examples produce better, more atomic questions
- **You need structured analysis** — Typed tuples (entity/attribute/relation) enable per-category breakdowns
- **You can absorb the cost** — Budget isn't a primary constraint

### The Hybrid Approach

The best solution might be combining both frameworks' strengths:

1. **DSG's question generation quality** — Use its 3-step pipeline (or a simplified version) to generate well-structured, dependency-aware questions
2. **Soft-TIFA's logprob scoring** — Score each question using token probabilities instead of hard yes/no
3. **Dependency-aware soft aggregation** — Apply DSG's dependency filtering to soft scores, but instead of zeroing out children, *discount* them by the parent's probability

This hybrid would look like:

```python
# Pseudocode for hybrid scoring
for qid, parent_ids in dependencies.items():
    parent_confidence = min(soft_scores[pid] for pid in parent_ids if pid != 0)
    adjusted_scores[qid] = soft_scores[qid] * parent_confidence
```

If the parent entity scores 0.3 (probably not there), the child attribute score gets multiplied by 0.3 instead of being hard-zeroed. This preserves both the structural awareness of DSG and the continuous signal of Soft-TIFA.

---

## Conclusion

Soft-TIFA and DSG represent two philosophies in T2I evaluation:

- **Soft-TIFA** prioritizes **scoring precision** — continuous logprobs over binary answers — at the cost of structural naiveté.
- **DSG** prioritizes **question structure** — dependency-aware, formally grounded decomposition — at the cost of scoring nuance and 6× higher expense.

Neither is perfect. Both suffer from the fundamental VLM bias problem (yes/no accuracy barely above chance). But for practical T2I evaluation today, **Soft-TIFA's logprob advantage is more impactful than DSG's dependency advantage**, because scoring precision affects every single question while dependency filtering only matters for the subset of questions where a parent entity is missing.

If you can afford it, the hybrid approach — DSG's question generation with Soft-TIFA's scoring — gives you the best of both worlds. If you need to pick one, **start with Soft-TIFA** and add dependency awareness later.

---

*This analysis is based on hands-on implementation of both frameworks in a production T2I grading system, using Azure OpenAI (GPT-4o) as the VLM backend. The source code for GenEval2 is available from Meta FAIR (CC BY-NC 4.0) and DSG from Google Research (ICLR 2024).*

*References:*
- *GenEval 2: "Addressing Benchmark Drift in Text-to-Image Evaluation" — Meta FAIR, 2024*
- *"Davidsonian Scene Graph: Improving Reliability in Fine-grained Evaluation for Text-to-Image Generation" — Cho et al., ICLR 2024*
- *"Leveraging Panoptic Scene Graph for Evaluating Fine-Grained Text-to-Image Generation" — Deng et al., ICCV 2025*
- *"TIFA: Accurate and Interpretable Text-to-Image Faithfulness Evaluation with Question Answering" — Hu et al., ICCV 2023*
