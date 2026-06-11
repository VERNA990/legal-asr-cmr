import os
import glob
import pandas as pd
from datasets import Dataset, Audio, DatasetDict
from transformers import WhisperProcessor

def load_and_transform_gated_data():
    csv_path = "/content/drive/MyDrive/fyp-phase0/phase0_dataset.csv"
    print(f"📊 Loading target evaluation data matrix from: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Missing critical database mapping index: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    print("🔍 Indexing actual physical files on disk for rapid mapping...")
    # Scan every single .wav file currently sitting in your extracted directories
    actual_wav_paths = glob.glob("/content/afrispeech/**/*.wav", recursive=True)
    
    # Create a lookup dictionary: { 'filename.wav': '/absolute/path/to/filename.wav' }
    disk_lookup = {os.path.basename(p): p for p in actual_wav_paths}
    print(f"📁 Indexed {len(disk_lookup)} physical audio tracks on runtime storage.")
    
    valid_paths = []
    valid_transcripts = []
    
    print("🛠️ Re-anchoring CSV records to physical disk paths...")
    missing_counter = 0
    
    for idx, row in df.iterrows():
        csv_path_entry = row['audio_path']
        transcript = row['transcript']
        
        # Extract just the filename (e.g., 'xyz.wav') from the long CSV path string
        filename = os.path.basename(csv_path_entry)
        
        # Check if that filename exists anywhere in our scanned disk lookup
        if filename in disk_lookup:
            path = disk_lookup[filename]
        elif os.path.exists(csv_path_entry):
            path = csv_path_entry
        else:
            missing_counter += 1
            continue  # Skip if the file physically isn't there
            
        if pd.isna(transcript) or not str(transcript).strip():
            continue
            
        valid_paths.append(path)
        valid_transcripts.append(str(transcript))
        
    print(f"🎵 Successfully verified and anchored {len(valid_paths)} audio channels for runtime execution.")
    if missing_counter > 0:
        print(f"⚠️ Skipped {missing_counter} rows because their filenames weren't found in /content/afrispeech.")
    
    if not valid_paths:
        raise ValueError("❌ Error: Zero active physical file matches found on disk using CSV mappings.")

    data_manifest = {
        "audio": valid_paths,
        "transcript": valid_transcripts
    }
    
    raw_dataset = Dataset.from_dict(data_manifest)
    
    print("✂️ Engineering train/validation/test evaluation boundaries (70/15/15)...")
    train_testval = raw_dataset.train_test_split(test_size=0.3, seed=42)
    test_val = train_testval["test"].train_test_split(test_size=0.5, seed=42)
    
    dataset_splits = DatasetDict({
        "train": train_testval["train"],
        "validation": test_val["train"],
        "test": test_val["test"]
    })
    
    print("🔄 Casting structural features to target pipeline audio formats (16kHz Mono)...")
    dataset_splits = dataset_splits.cast_column("audio", Audio(sampling_rate=16000))
    
    processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="English", task="transcribe")
    
    def transform_function(batch):
        audio_sample = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio_sample["array"], 
            sampling_rate=audio_sample["sampling_rate"]
        ).input_features[0]
        
        batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
        return batch

    print("⚡ Mapping Whisper feature tokenizations across matrix splits...")
    processed_dataset = dataset_splits.map(
        transform_function, 
        remove_columns=["audio", "transcript"], 
        num_proc=1
    )
    
    print("✅ Local alignment data components finalized successfully!")
    return processed_dataset, processor
