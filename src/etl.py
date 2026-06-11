import os
from datasets import load_dataset, Audio, DatasetDict
from transformers import WhisperProcessor
from google.colab import userdata

def load_and_transform_gated_data():
    print("🔑 Fetching secure Hugging Face token from Colab Secrets...")
    try:
        hf_token = userdata.get('HF_TOKEN')
    except Exception as e:
        raise ValueError("HF_TOKEN not found in Colab Secrets. Please add it via the Key icon sidebar.")

    print("📥 Attempting download of gated 'intronhealth/afrispeech-countries' dataset...")
    # Passing the token parameter directly to authorize the download
    raw_dataset = load_dataset(
        "intronhealth/afrispeech-countries",
        token=hf_token
    )

    # Note: 'afrispeech-countries' is a benchmark dataset without native train splits.
    # We enforce our own strict 70/15/15 split if a single split is loaded.
    if isinstance(raw_dataset, DatasetDict):
        dataset_to_split = raw_dataset[list(raw_dataset.keys())[0]]
    else:
        dataset_to_split = raw_dataset

    print("✂️ Creating strict 70/15/15 structural data splits...")
    train_testval = dataset_to_split.train_test_split(test_size=0.3, seed=42)
    test_val = train_testval["test"].train_test_split(test_size=0.5, seed=42)

    dataset_splits = DatasetDict({
        "train": train_testval["train"],
        "validation": test_val["train"],
        "test": test_val["test"]
    })

    print("🔄 Forcing acoustic resampling to 16,000Hz mono...")
    dataset_splits = dataset_splits.cast_column("audio", Audio(sampling_rate=16000))

    processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="English", task="transcribe")

    def transform_function(batch):
        audio_sample = batch["audio"]
        # Convert audio waveform arrays into an 80-channel log-Mel spectrogram feature map
        batch["input_features"] = processor.feature_extractor(
            audio_sample["array"],
            sampling_rate=audio_sample["sampling_rate"]
        ).input_features[0]

        # Tokenize target text transcript strings to IDs for the Whisper decoder
        batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
        return batch

    print("⚡ Mapping transformations across splits via multiprocessing...")
    processed_dataset = dataset_splits.map(
        transform_function,
        remove_columns=dataset_to_split.column_names,
        num_proc=2
    )

    print("✅ Gated dataset loaded and prepared successfully!")
    return processed_dataset, processor
