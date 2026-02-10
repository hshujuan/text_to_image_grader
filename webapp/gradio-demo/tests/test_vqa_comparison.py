"""
Test script to compare VQA pair generation methods:
1. GenEval2 Original (from geneval2_data.jsonl)
2. Rule-Based Extraction
3. LLM Fallback (GPT-4o)

This helps validate that our rule-based approach matches GenEval2's methodology.
"""

import sys
import json
import os
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()
GRADIO_DEMO_DIR = SCRIPT_DIR.parent

# Add src to path (relative to script location)
sys.path.insert(0, str(GRADIO_DEMO_DIR / 'src'))

from metrics.soft_tifa import _extract_vqa_list_rulebased, _extract_vqa_list_llm

# For LLM testing, we need the Azure OpenAI client
try:
    from openai import AzureOpenAI
    from dotenv import load_dotenv
    
    # Load .env from gradio-demo directory
    load_dotenv(GRADIO_DEMO_DIR / '.env')
    
    # Try grading endpoint first, then fall back to main endpoint
    grading_endpoint = os.getenv("AZURE_OPENAI_GRADING_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
    grading_key = os.getenv("AZURE_OPENAI_GRADING_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    grading_deployment = os.getenv("AZURE_OPENAI_GRADING_DEPLOYMENT", "gpt-4o")
    
    if grading_endpoint and grading_key:
        grading_client = AzureOpenAI(
            azure_endpoint=grading_endpoint,
            api_key=grading_key,
            api_version="2024-02-15-preview"
        )
        LLM_AVAILABLE = True
        print(f"✅ Azure OpenAI configured: {grading_deployment}")
    else:
        LLM_AVAILABLE = False
        print("⚠️  Azure OpenAI not configured. Skipping LLM comparison.")
        print(f"   Endpoint: {'✓' if grading_endpoint else '✗'}")
        print(f"   Key: {'✓' if grading_key else '✗'}")
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️  openai package not available. Skipping LLM comparison.")


def load_geneval2_samples(n=5, start_idx=0):
    """Load n samples from geneval2_data.jsonl starting at start_idx"""
    # Try multiple possible paths relative to script location
    # SCRIPT_DIR is tests/, GRADIO_DEMO_DIR is gradio-demo/
    possible_paths = [
        GRADIO_DEMO_DIR.parent.parent / "lib" / "geneval2" / "geneval2_data.jsonl",
        GRADIO_DEMO_DIR / ".." / ".." / "lib" / "geneval2" / "geneval2_data.jsonl",
        Path("lib/geneval2/geneval2_data.jsonl"),
    ]
    
    data_path = None
    for p in possible_paths:
        if p.exists():
            data_path = p
            break
    
    if not data_path:
        print(f"❌ Could not find geneval2_data.jsonl")
        print(f"   Tried: {[str(p) for p in possible_paths]}")
        return []
    
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < start_idx:
                continue
            if len(samples) >= n:
                break
            samples.append(json.loads(line))
    
    return samples


def format_vqa_list(vqa_list, skills=None):
    """Format VQA list for display"""
    lines = []
    for i, (q, a) in enumerate(vqa_list):
        skill = f"[{skills[i]}]" if skills and i < len(skills) else ""
        lines.append(f"    {skill:12} Q: {q}")
        lines.append(f"              A: {a}")
    return "\n".join(lines)


def compare_vqa_lists(original, generated):
    """Compare two VQA lists and return match statistics (order-independent)"""
    # Normalize: lowercase and strip whitespace
    original_set = set((q.lower().strip(), a.lower().strip()) for q, a in original)
    generated_set = set((q.lower().strip(), a.lower().strip()) for q, a in generated)
    
    matches = original_set & generated_set
    missing = original_set - generated_set
    extra = generated_set - original_set
    
    return {
        'match_count': len(matches),
        'original_count': len(original),
        'generated_count': len(generated),
        'matches': matches,
        'missing': missing,
        'extra': extra,
        'match_pct': len(matches) / len(original) * 100 if original else 0
    }


def main():
    print("=" * 80)
    print("VQA PAIR GENERATION COMPARISON")
    print("Comparing: GenEval2 Original vs Rule-Based vs LLM Fallback")
    print("Using complex prompts (796-800) with 10 atoms each")
    print("=" * 80)
    
    # Load complex prompts from the end of the file (796-800, 0-indexed: 795-799)
    samples = load_geneval2_samples(n=5, start_idx=795)
    
    rule_based_stats = {'total_original': 0, 'total_matches': 0}
    llm_stats = {'total_original': 0, 'total_matches': 0}
    
    for i, sample in enumerate(samples):
        prompt = sample['prompt']
        original_vqa = sample['vqa_list']
        original_skills = sample.get('skills', [])
        
        print(f"\n{'─' * 80}")
        print(f"EXAMPLE {i+1}: \"{prompt}\"")
        print(f"Atom Count: {sample.get('atom_count', 'N/A')}")
        print(f"{'─' * 80}")
        
        # 1. Original GenEval2
        print(f"\n📋 GENEVAL2 ORIGINAL ({len(original_vqa)} pairs):")
        print(format_vqa_list(original_vqa, original_skills))
        
        # 2. Rule-Based
        rule_vqa, rule_skills = _extract_vqa_list_rulebased(prompt)
        print(f"\n🔧 RULE-BASED ({len(rule_vqa)} pairs):")
        print(format_vqa_list(rule_vqa, rule_skills))
        
        # Compare rule-based
        rule_comparison = compare_vqa_lists(original_vqa, rule_vqa)
        rule_based_stats['total_original'] += rule_comparison['original_count']
        rule_based_stats['total_matches'] += rule_comparison['match_count']
        print(f"\n   📊 Match: {rule_comparison['match_count']}/{rule_comparison['original_count']} ({rule_comparison['match_pct']:.1f}%) [order-independent]")
        if rule_comparison['missing']:
            print(f"   ❌ Missing from rule-based:")
            for q, a in list(rule_comparison['missing'])[:3]:  # Show first 3
                print(f"      - \"{q}\" → {a}")
        
        # 3. LLM Fallback (if available)
        if LLM_AVAILABLE:
            print(f"\n🤖 LLM FALLBACK (GPT-4o):")
            try:
                llm_vqa = _extract_vqa_list_llm(prompt, grading_client, grading_deployment)
                print(format_vqa_list(llm_vqa))
                
                llm_comparison = compare_vqa_lists(original_vqa, llm_vqa)
                llm_stats['total_original'] += llm_comparison['original_count']
                llm_stats['total_matches'] += llm_comparison['match_count']
                print(f"\n   📊 Match: {llm_comparison['match_count']}/{llm_comparison['original_count']} ({llm_comparison['match_pct']:.1f}%) [order-independent]")
                if llm_comparison['missing']:
                    print(f"   ❌ Missing (different phrasing):")
                    for q, a in list(llm_comparison['missing'])[:3]:  # Show first 3
                        print(f"      - \"{q}\" → {a}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            print(f"\n🤖 LLM FALLBACK: Skipped (not configured)")
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    
    if rule_based_stats['total_original'] > 0:
        overall_rule_pct = rule_based_stats['total_matches'] / rule_based_stats['total_original'] * 100
        print(f"\n🔧 Rule-Based Overall Match: {rule_based_stats['total_matches']}/{rule_based_stats['total_original']} ({overall_rule_pct:.1f}%)")
    
    if LLM_AVAILABLE and llm_stats['total_original'] > 0:
        overall_llm_pct = llm_stats['total_matches'] / llm_stats['total_original'] * 100
        print(f"🤖 LLM Overall Match: {llm_stats['total_matches']}/{llm_stats['total_original']} ({overall_llm_pct:.1f}%)")


if __name__ == "__main__":
    main()
