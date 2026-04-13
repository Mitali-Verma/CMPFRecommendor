import gzip
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Detect available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure model runs in evaluation mode (no gradients)
torch.set_grad_enabled(False)

# Optimizations for whatever device is available
if device.type == 'cuda':
    print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
    batch_size = 128  # Larger batches on GPU
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use up to 80% of GPU memory
else:
    print("💻 Using CPU")
    batch_size = 64   # Smaller batches on CPU
    torch.set_num_threads(8)  # Use multiple CPU threads

# Load model and tokenizer
print("=" * 70)
print("PHASE 3: Generate DistilBERT Embeddings with Mean Pooling")
print("=" * 70)

print(f"\nDevice: {device}")
print(f"Batch size: {batch_size}")
print("\nLoading DistilBERT model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased')
model.eval()  # Set to evaluation mode (frozen, no dropout)
model.to(device)  # Move model to device

print(f"✓ Model loaded: {model.__class__.__name__}")
print(f"✓ Embedding dimension: 768")

# Load truncated text strings
print("\nLoading truncated text strings...")
texts_dict = {}
with open('item_texts_truncated.txt', 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc="Reading text file"):
        parts = line.strip().split('\t', 1)
        if len(parts) == 2:
            asin, text = parts
            texts_dict[asin] = text

print(f"✓ Loaded {len(texts_dict)} text strings")

# Create asin-to-index mapping
asin_list = list(texts_dict.keys())
asin_to_index = {asin: idx for idx, asin in enumerate(asin_list)}

print(f"✓ Created ASIN-to-index mapping")

# Generate embeddings with mean pooling
embeddings_list = []

print(f"\nGenerating embeddings with mean pooling (batch_size={batch_size})...")

for i in tqdm(range(0, len(asin_list), batch_size), desc="Embedding batches"):
    batch_asins = asin_list[i:i+batch_size]
    batch_texts = [texts_dict[asin] for asin in batch_asins]
    
    # Tokenize batch
    inputs = tokenizer(
        batch_texts, 
        return_tensors='pt', 
        max_length=512,
        padding=True, 
        truncation=True
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate embeddings (no gradients)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # outputs.last_hidden_state shape: (batch_size, seq_len, 768)
    last_hidden_state = outputs.last_hidden_state
    
    # Create attention mask to ignore padding tokens
    attention_mask = inputs['attention_mask'].unsqueeze(-1)  # (batch_size, seq_len, 1)
    
    # Mean pooling: sum embeddings weighted by attention mask, then divide by sequence length
    masked_embeddings = last_hidden_state * attention_mask  # Zero out padding
    sum_embeddings = torch.sum(masked_embeddings, dim=1)     # (batch_size, 768)
    sum_mask = torch.sum(attention_mask, dim=1)              # (batch_size, 1)
    
    mean_pooled = sum_embeddings / sum_mask                  # (batch_size, 768)
    
    # Convert to numpy and append (move back to CPU if on GPU)
    embeddings_list.append(mean_pooled.cpu().numpy())

# Stack all embeddings
embeddings_array = np.vstack(embeddings_list)

print(f"\n✓ Generated embeddings shape: {embeddings_array.shape}")
print(f"  Expected shape: ({len(asin_list)}, 768)")

# Verify shape
assert embeddings_array.shape == (len(asin_list), 768), \
    f"Embedding shape mismatch: {embeddings_array.shape} vs ({len(asin_list)}, 768)"

# Save embeddings
print("\nSaving embeddings...")
np.save('item_embeddings_v2.npy', embeddings_array)
print(f"✓ Saved to item_embeddings_v2.npy")

# Save ASIN-to-index mapping
print("Saving ASIN-to-index mapping...")
with open('asin_to_index.json', 'w', encoding='utf-8') as f:
    json.dump(asin_to_index, f, indent=2)
print(f"✓ Saved to asin_to_index.json ({len(asin_to_index)} items)")

# Sample statistics
print(f"\nEmbedding Statistics:")
print(f"  Mean norm of embedding vectors: {np.linalg.norm(embeddings_array, axis=1).mean():.4f}")
print(f"  Std dev of embedding norms: {np.linalg.norm(embeddings_array, axis=1).std():.4f}")
print(f"  Mean value across all dimensions: {embeddings_array.mean():.6f}")
print(f"  Std dev across all dimensions: {embeddings_array.std():.6f}")

print("\n" + "=" * 70)
print("PHASE 3 COMPLETE")
print("=" * 70)
print(f"\nDevice used: {device}")
print("\nOutput files:")
print("  1. item_embeddings_v2.npy - Shape (164817, 768)")
print("  2. asin_to_index.json - ASIN → index mapping")
print("\nTo load:")
print("  embeddings = np.load('item_embeddings_v2.npy')")
print("  with open('asin_to_index.json') as f: asin_to_index = json.load(f)")
