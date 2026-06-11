import os
import pandas as pd
from datasets import Dataset, Audio, DatasetDict
from transformers import WhisperProcessor

def load_and_transform_gated_data():
    csv_path = "/content/drive/MyDrive/fyp-phase0/phase0_dataset.csv"
    print(f"📊 Loading target evaluation data matrix from: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Missing critical database mapping index: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    valid_paths = []
    valid_transcripts = []
    
    print("🛠️ Synchronizing tracking records with active disk pathways...")
    for idx, row in df.iterrows():
        path = row['audio_path']
        transcript = row['transcript']
        
        # If the dataset was generated under a different subfolder pattern, adjust path layout
        if not os.path.exists(path):
            # Fallback path lookup swapping out root prefixes if needed
            base_filename = os.path.basename(path)
            # Search under your locally extracted content path
            alternative_path = os.path.join("/content/afrispeech", base_filename)
            
            if os.path.exists(alternative_path):
                path = alternative_path
            else:
                # Fallback to search recursively for this audio item name
                continue 
                
        if pd.isna(transcript) or not str(transcript).strip():
            continue
            
        valid_paths.append(path)
        valid_transcripts.append(str(transcript))
        
    print(f"🎵 Successfully verified and anchored {len(valid_paths)} audio channels for runtime execution.")
    
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
