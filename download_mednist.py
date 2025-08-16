#!/usr/bin/env python3
"""
📥 Download MedNIST Dataset to VM
This script downloads the MedNIST dataset for GPU processing
"""

import os
import urllib.request
import hashlib
import tarfile
import shutil

def download_and_extract(url, compressed_file, data_dir, md5_hash):
    """Download and extract the dataset"""
    
    print(f"📥 Downloading MedNIST dataset from: {url}")
    print(f"💾 Saving to: {compressed_file}")
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Download file
    urllib.request.urlretrieve(url, compressed_file)
    
    # Verify MD5 hash
    print("🔍 Verifying download integrity...")
    with open(compressed_file, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    
    if file_hash != md5_hash:
        raise ValueError(f"MD5 hash mismatch! Expected: {md5_hash}, Got: {file_hash}")
    
    print("✅ Download integrity verified!")
    
    # Extract dataset
    print("📦 Extracting dataset...")
    with tarfile.open(compressed_file, 'r:gz') as tar:
        tar.extractall(data_dir)
    
    print("✅ Dataset extracted successfully!")
    
    # Clean up compressed file
    os.remove(compressed_file)
    print("🧹 Compressed file removed")
    
    # List contents
    print("\n📁 Dataset contents:")
    for root, dirs, files in os.walk(data_dir):
        level = root.replace(data_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")

def main():
    """Main execution function"""
    print("🚀 MedNIST Dataset Downloader")
    print("=" * 40)
    
    # Configuration
    data_dir = "./mednist"
    resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
    md5 = "0bc7306e7427e00ad1c5526a6677552d"
    
    compressed_file = os.path.join(data_dir, "MedNIST.tar.gz")
    
    # Check if dataset already exists
    if os.path.exists(os.path.join(data_dir, "MedNIST")):
        print("✅ MedNIST dataset already exists!")
        print(f"📁 Location: {os.path.abspath(data_dir)}")
        return
    
    try:
        # Download and extract
        download_and_extract(resource, compressed_file, data_dir, md5)
        
        print(f"\n🎉 MedNIST dataset downloaded successfully!")
        print(f"📁 Dataset location: {os.path.abspath(data_dir)}")
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        raise

if __name__ == "__main__":
    main()
