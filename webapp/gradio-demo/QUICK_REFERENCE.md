# 📊 Quick Metrics Reference Card

## At a Glance

| Metric | Type | What It Measures | Good Score |
|--------|------|------------------|------------|
| **Soft-TIFA GM** | 🔍 VLM | Does image contain all prompt facts? | 80+ |
| **VQAScore** | 🤖 Model | Can VQA model answer questions correctly? | 70+ |
| **CLIPScore** | 🤖 Model | Text-image embedding similarity | 70+ |
| **CMMD** | 🤖 Model | Cross-modal distance | 60+ |
| **AHEaD** | 🤖 Model | Attention-based alignment | 60+ |
| **PickScore** | 🤖 Model | Human preference estimate | 70+ |
| **BRISQUE** | 📐 Code | Spatial quality (blur, artifacts) | 80+ |
| **NIQE** | 📐 Code | Natural image quality | 80+ |
| **CLIP-IQA** | 🤖 Model | Visual quality assessment | 70+ |
| **Toxicity** | 🔍 VLM | No harmful content | 95+ |
| **Fairness** | 🔍 VLM | No bias/stereotypes | 95+ |
| **Privacy** | 🔍 VLM | No personal info | 95+ |

---

## Icon Legend

- 🤖 **Model** = Runs an ML model (CLIP, ViLT)
- 📐 **Code** = Mathematical algorithm (no ML)
- 🔍 **VLM** = GPT-4o analyzes visually

---

## Which Metrics to Trust Most?

### For "Did it match my prompt?"
1. **Soft-TIFA GM** ⭐ (most thorough)
2. **VQAScore** (direct Q&A)
3. **CLIPScore** (industry standard)

### For "Does it look good?"
1. **Qualitative Assessment** (human-like)
2. **CLIP-IQA** (learned quality)
3. **BRISQUE/NIQE** (technical issues)

### For "Is it safe/ethical?"
- Check **all three** safety metrics
- Review specific issues listed

---

## Common Score Patterns

| Pattern | Meaning |
|---------|---------|
| High Soft-TIFA, Low Quality | Got the content right, but looks bad |
| Low Soft-TIFA, High Quality | Beautiful but wrong content |
| High CLIP, Low VQA | Superficial match, missing details |
| Low Safety (any) | **Review immediately** |

---

## Report Sections Order

1. **Legend** - Icon meanings
2. **T2ISafety** - Safety check results
3. **Soft-TIFA** - Atomic fact breakdown
4. **Comprehensive Scores** - All metrics
   - North Star (Soft-TIFA)
   - Alignment (5 metrics)
   - Quality (3 metrics)
   - Safety (3 metrics)
   - Qualitative (GPT-4o text)
5. **Performance** - Timing info

---

**📖 For detailed explanations, see [METRICS_GUIDE.md](METRICS_GUIDE.md)**
