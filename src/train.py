import torch
import argparse
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model
from etl import load_and_transform_gated_data
from collator import DataCollatorSpeechSeq2SeqWithPadding

def run_training():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Path to Google Drive checkpoint directory")
    args = parser.parse_args()

    # 1. Run the unified ETL engine
    processed_dataset, processor = load_and_transform_gated_data()
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 2. Load the base whisper-medium model weights in half-precision (float16) to conserve VRAM
    model_id = "openai/whisper-medium"
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 3. Inject parameter-efficient LoRA adapters targeting cross-attention projections
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Set up hyperparameter metrics for Supervised Fine-Tuning (SFT)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_steps=100,
        max_steps=2000,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=4,
        eval_steps=250,
        save_steps=250,
        logging_steps=50,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )

    print("🔥 Starting Supervised Fine-Tuning Loop...")
    trainer.train()
    print("🏆 Fine-Tuning sequence complete!")

if __name__ == "__main__":
    run_training()
