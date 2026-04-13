import gzip
import json

# Function to peek into a gz file
def peek_gz_file(file_path, num_lines=5):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break
            print(json.dumps(json.loads(line), indent=2))
            print("---")

# Peek into both files
print("Peeking into Handmade_Products.jsonl.gz:")
peek_gz_file('Handmade_Products.jsonl.gz')

print("\nPeeking into meta_Handmade_Products.jsonl.gz:")
peek_gz_file('meta_Handmade_Products.jsonl.gz')