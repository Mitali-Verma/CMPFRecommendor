import gzip
import json
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def extract_material_and_technique(item):
    """
    Extract material and technique from available fields.
    These might be in title, description, or features.
    """
    material = None
    technique = None
    
    # Common materials to look for
    materials_keywords = ['copper', 'silver', 'gold', 'wood', 'leather', 'steel', 'ceramic', 
                         'glass', 'fabric', 'clay', 'bronze', 'aluminum', 'titanium', 'plastic']
    
    # Common techniques to look for
    technique_keywords = ['hand-stamped', 'hand-painted', 'etched', 'carved', 'woven', 
                         'knitted', 'crocheted', 'stitched', 'embroidered', 'molded', 
                         'cast', 'forged', 'hammered', 'soldered', 'glazed']
    
    text_to_search = (item.get('title', '') + ' ' + 
                     ' '.join(item.get('description', [])) + ' ' + 
                     ' '.join(item.get('features', []))).lower()
    
    for mat in materials_keywords:
        if mat in text_to_search:
            material = mat
            break
    
    for tech in technique_keywords:
        if tech in text_to_search:
            technique = tech
            break
    
    return material, technique

def build_text_string(item):
    """
    Build concatenated text string for an item with specific format.
    Format: [TITLE]: {title}. [MATERIAL]: {material}. [TECHNIQUE]: {technique}. 
             [DESCRIPTION]: {first 200 chars}. [FEATURES]: {features joined}
    """
    title = item.get('title', '')
    material, technique = extract_material_and_technique(item)
    description = ' '.join(item.get('description', []))[:200]  # First 200 chars
    features = ' '.join(item.get('features', []))
    
    parts = [f"[TITLE]: {title}"]
    
    if material:
        parts.append(f"[MATERIAL]: {material}")
    
    if technique:
        parts.append(f"[TECHNIQUE]: {technique}")
    
    if description:
        parts.append(f"[DESCRIPTION]: {description}")
    
    if features:
        parts.append(f"[FEATURES]: {features}")
    
    text_string = ". ".join(parts)
    return text_string

def load_meta_and_build_texts(file_path):
    """
    Load metadata and build text strings for all items.
    """
    data = {}
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading and building text strings"):
            item = json.loads(line)
            asin = item.get('parent_asin')
            if asin:
                text_string = build_text_string(item)
                data[asin] = text_string
    return data

def tokenize_and_truncate(texts_dict, max_tokens=400):
    """
    Tokenize all text strings and truncate to max_tokens.
    Returns tokenized strings and token counts.
    """
    truncated_texts = {}
    token_counts = []
    
    for asin, text in tqdm(texts_dict.items(), desc="Tokenizing and truncating"):
        # Tokenize
        tokens = tokenizer.encode(text, truncation=False)
        token_count = len(tokens)
        token_counts.append(token_count)
        
        # Truncate if necessary
        if token_count > max_tokens:
            truncated_tokens = tokens[:max_tokens]
            truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        else:
            truncated_text = text
        
        truncated_texts[asin] = truncated_text
    
    return truncated_texts, token_counts

# Main pipeline
print("=" * 70)
print("PHASE 2: Building Handmade-specific Text Strings")
print("=" * 70)

meta_file = 'meta_Handmade_Products.jsonl.gz'

# Load and build text strings
print("\nStep 1: Building text strings...")
texts_dict = load_meta_and_build_texts(meta_file)
print(f"Built text strings for {len(texts_dict)} items")

# Show sample
print("\nSample text strings:")
for i, (asin, text) in enumerate(list(texts_dict.items())[:3]):
    print(f"\n  ASIN {i}: {asin}")
    print(f"  Text: {text[:150]}...")

# Tokenize and truncate
print("\nStep 2: Tokenizing and truncating to 400 tokens...")
truncated_texts, token_counts = tokenize_and_truncate(texts_dict, max_tokens=400)

# Statistics
import numpy as np
token_counts = np.array(token_counts)
print(f"\nTokenization Statistics:")
print(f"  Mean tokens: {token_counts.mean():.2f}")
print(f"  Median tokens: {np.median(token_counts):.2f}")
print(f"  Max tokens: {token_counts.max()}")
print(f"  Min tokens: {token_counts.min()}")
print(f"  Items truncated: {(token_counts > 400).sum()}")

# Save truncated texts
print("\nStep 3: Saving truncated text strings...")
with open('item_texts_truncated.txt', 'w', encoding='utf-8') as f:
    for asin in sorted(truncated_texts.keys()):
        f.write(f"{asin}\t{truncated_texts[asin]}\n")

print("Saved to item_texts_truncated.txt")

print("\n" + "=" * 70)
print("PHASE 2 COMPLETE")
print("=" * 70)
