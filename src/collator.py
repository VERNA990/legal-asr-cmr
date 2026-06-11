import torch
import dataclasses
from typing import Any, Dict, List, Union

@dataclasses.dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 1. Grab raw audio input arrays and pad dynamically to the longest element in the batch
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # 2. Grab text sequences and process padding dimensions
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 3. Mask pad tokens with -100 so the Cross-Entropy Loss engine skips evaluating them
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )
        
        # 4. Anchor our clean sequence arrays to the labels channel
        batch["labels"] = labels

        # 5. Hard isolation boundary: Safely pop 'input_ids' completely out of the batch dictionary
        if "input_ids" in batch:
            del batch["input_ids"]

        return batch
