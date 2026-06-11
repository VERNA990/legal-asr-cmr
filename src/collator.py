import torch
import dataclasses
from typing import Any, Dict, List, Union

@dataclasses.dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 1. Isolate and pad the raw acoustic input features (spectrogram arrays)
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # 2. Isolate and pad target token strings under the explicit tokenizer sequence key
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 3. Cleanly mask padding token IDs with -100 so Cross-Entropy loss ignores them
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )
        
        # 4. Explicitly bind only the clean labels to the batch execution matrix
        batch["labels"] = labels

        # Sanity Check: If an unintended 'input_ids' key leaked into batch from padding, pop it out
        if "input_ids" in batch:
            del batch["input_ids"]

        return batch
