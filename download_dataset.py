import os
import zipfile
import urllib.request
import pandas as pd
from tqdm import tqdm

class TqdmUpTo(tqdm):
    """Provides progress bar for urllib.request.urlretrieve."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=os.path.basename(dest_path)) as t:
        urllib.request.urlretrieve(url, filename=dest_path, reporthook=t.update_to)

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    flickr_dir = os.path.join(data_dir, "flickr8k")
    
    os.makedirs(flickr_dir, exist_ok=True)
    
    # Mirror URLs for Flickr8k (Jason Brownlee's release)
    images_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
    captions_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
    
    images_zip = os.path.join(flickr_dir, "Flickr8k_Dataset.zip")
    captions_zip = os.path.join(flickr_dir, "Flickr8k_text.zip")
    
    # Download files if they don't exist
    if not os.path.exists(images_zip):
        download_file(images_url, images_zip)
    else:
        print("Images zip already downloaded.")
        
    if not os.path.exists(captions_zip):
        download_file(captions_url, captions_zip)
    else:
        print("Captions zip already downloaded.")
        
    # Extract images
    images_dest = os.path.join(flickr_dir, "Images")
    if not os.path.exists(images_dest):
        print("Extracting images...")
        with zipfile.ZipFile(images_zip, 'r') as zip_ref:
            # We want to extract to flickr_dir. 
            # Note: The zip contains a folder called "Flicker8k_Dataset" (notice the spelling difference 'Flicker' vs 'Flickr')
            zip_ref.extractall(flickr_dir)
        
        # Rename the extracted folder "Flicker8k_Dataset" to "Images"
        old_folder = os.path.join(flickr_dir, "Flicker8k_Dataset")
        if os.path.exists(old_folder):
            os.rename(old_folder, images_dest)
            print("Renamed extracted folder to Images.")
    else:
        print("Images already extracted.")
        
    # Extract captions
    captions_txt_extracted = os.path.join(flickr_dir, "Flickr8k.token.txt")
    if not os.path.exists(captions_txt_extracted):
        print("Extracting captions text...")
        with zipfile.ZipFile(captions_zip, 'r') as zip_ref:
            zip_ref.extractall(flickr_dir)
    else:
        print("Captions already extracted.")
        
    # Parse the raw tokens file into a clean CSV captions.txt
    output_captions = os.path.join(flickr_dir, "captions.txt")
    if not os.path.exists(output_captions):
        print("Parsing captions into clean CSV format...")
        data = []
        with open(captions_txt_extracted, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                img_id, caption = parts[0], parts[1]
                # Image id format: 1000268201_693b08cb0e.jpg#0
                img_name = img_id.split('#')[0]
                data.append({"image": img_name, "caption": caption})
        
        df = pd.DataFrame(data)
        df.to_csv(output_captions, index=False)
        print(f"Created clean captions file at {output_captions} with {len(df)} rows.")
    else:
        print("Clean captions.txt already exists.")
        
    print("Dataset setup completed successfully!")

if __name__ == "__main__":
    main()
