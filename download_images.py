import gzip
import json
import os
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create output directory
OUTPUT_DIR = 'data/images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
IMAGE_SIZE = (224, 224)
MAX_WORKERS = 10
TIMEOUT = 10
FAILED_LOG = 'failed_images.txt'

# Statistics
stats = {
    'total': 0,
    'downloaded': 0,
    'failed': 0,
    'missing': 0,
    'placeholder_created': 0
}

def load_metadata(file_path):
    """Load metadata from gzipped JSONL file."""
    data = {}
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading metadata"):
            item = json.loads(line)
            asin = item.get('parent_asin')
            images = item.get('images', [])
            if asin and images:
                # Extract the first (main) image URL
                first_image = images[0]
                if isinstance(first_image, dict):
                    # Try different URL keys based on metadata structure
                    url = first_image.get('large') or first_image.get('hi_res')
                else:
                    url = None
                
                if url:
                    data[asin] = url
    return data

def download_image(asin, url):
    """Download and resize a single image."""
    try:
        # Download image
        response = requests.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        
        # Open image
        img = Image.open(BytesIO(response.content))
        
        # Convert to RGB if necessary (handles RGBA, greyscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to 224x224
        img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        # Save
        output_path = os.path.join(OUTPUT_DIR, f'{asin}.jpg')
        img.save(output_path, 'JPEG', quality=95)
        
        return {'status': 'success', 'asin': asin, 'path': output_path}
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning(f"404 Not Found: {asin} - {url}")
        else:
            logger.warning(f"HTTP Error {e.response.status_code}: {asin} - {url}")
        return {'status': 'failed', 'asin': asin, 'error': str(e), 'type': 'http_error'}
    
    except Exception as e:
        logger.error(f"Failed to download {asin}: {str(e)}")
        return {'status': 'failed', 'asin': asin, 'error': str(e), 'type': 'other_error'}

def create_placeholder(asin):
    """Create a grey placeholder 224x224 image for missing items."""
    try:
        # Create grey image
        grey_img = Image.new('RGB', IMAGE_SIZE, color=(128, 128, 128))
        
        # Save as missing image
        output_path = os.path.join(OUTPUT_DIR, f'{asin}_missing.jpg')
        grey_img.save(output_path, 'JPEG', quality=95)
        
        return output_path
    except Exception as e:
        logger.error(f"Failed to create placeholder for {asin}: {str(e)}")
        return None

def download_images_parallel(asin_url_dict, max_workers=MAX_WORKERS):
    """Download images in parallel using ThreadPoolExecutor."""
    failed_asins = []
    
    print(f"\n" + "=" * 70)
    print("Downloading Images in Parallel")
    print("=" * 70)
    print(f"Total items: {len(asin_url_dict)}")
    print(f"Max workers: {max_workers}\n")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(download_image, asin, url): asin 
            for asin, url in asin_url_dict.items()
        }
        
        # Process completed tasks as they finish
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            result = future.result()
            asin = result['asin']
            
            if result['status'] == 'success':
                stats['downloaded'] += 1
            else:
                stats['failed'] += 1
                failed_asins.append((asin, result.get('error', 'Unknown error')))
    
    # Log failed items
    if failed_asins:
        logger.info(f"\n{len(failed_asins)} items failed to download. Creating placeholders...")
        with open(FAILED_LOG, 'w') as f:
            for asin, error in failed_asins:
                f.write(f"{asin}\t{error}\n")
        
        # Create placeholders for failed items
        for asin, _ in failed_asins:
            if create_placeholder(asin):
                stats['placeholder_created'] += 1
            else:
                stats['missing'] += 1
    
    stats['total'] = len(asin_url_dict)
    
    return failed_asins

def verify_images():
    """Verify downloaded images and check for corrupted files."""
    print("\n" + "=" * 70)
    print("Verifying Downloaded Images")
    print("=" * 70)
    
    image_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg')]
    corrupted = []
    
    for filename in tqdm(image_files, desc="Verifying"):
        try:
            img_path = os.path.join(OUTPUT_DIR, filename)
            with Image.open(img_path) as img:
                # Check size
                if img.size != IMAGE_SIZE:
                    logger.warning(f"Wrong size: {filename} - {img.size}")
                    corrupted.append(filename)
        except Exception as e:
            logger.error(f"Corrupted image: {filename} - {str(e)}")
            corrupted.append(filename)
    
    if corrupted:
        logger.warning(f"Found {len(corrupted)} corrupted/wrong-size images")
    else:
        logger.info(f"✓ All {len(image_files)} images verified successfully!")
    
    return corrupted

# Main execution
if __name__ == '__main__':
    print("=" * 70)
    print("PHASE 4: Download and Process Product Images")
    print("=" * 70)
    
    # Load metadata
    print("\nStep 1: Loading metadata...")
    meta_file = 'meta_Handmade_Products.jsonl.gz'
    asin_url_dict = load_metadata(meta_file)
    print(f"\n✓ Loaded {len(asin_url_dict)} items with images")
    
    # Download images in parallel
    print("\nStep 2: Downloading images in parallel...")
    failed_asins = download_images_parallel(asin_url_dict, max_workers=MAX_WORKERS)
    
    # Create placeholders for items with no image
    print("\nStep 3: Creating placeholders for items without images...")
    
    # Load full metadata to find items with no images
    all_asins = set()
    with gzip.open(meta_file, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="Finding items without images"):
            item = json.loads(line)
            asin = item.get('parent_asin')
            if asin:
                all_asins.add(asin)
    
    items_without_images = all_asins - set(asin_url_dict.keys())
    print(f"\nItems without images: {len(items_without_images)}")
    
    for asin in tqdm(items_without_images, desc="Creating placeholders"):
        if create_placeholder(asin):
            stats['placeholder_created'] += 1
        else:
            stats['missing'] += 1
    
    # Verify images
    print("\nStep 4: Verifying images...")
    corrupted = verify_images()
    
    # Final statistics
    print("\n" + "=" * 70)
    print("PHASE 4 SUMMARY")
    print("=" * 70)
    print(f"Total products: {len(all_asins)}")
    print(f"Successfully downloaded: {stats['downloaded']}")
    print(f"Failed downloads: {stats['failed']}")
    print(f"Placeholders created: {stats['placeholder_created']}")
    print(f"Missing/error placeholders: {stats['missing']}")
    print(f"Success rate: {100 * stats['downloaded'] / len(asin_url_dict):.1f}%")
    print(f"Images with issues: {len(corrupted)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Failed log: {FAILED_LOG}")
    print("=" * 70)
