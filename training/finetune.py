import json
import logging
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./recipe-bot-finetuned"
DATASET_FILE = "recipes_training.json"


def load_and_prepare_data(file_path):
    with open(file_path) as f:
        data = json.load(f)

    return Dataset.from_dict({"text": [item["text"] for item in data]})


def create_qlora_config():
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def tokenize_function(examples, tokenizer, max_length=512):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )


def main():
    logger.info("Recipe Bot QLoRA Fine-tuning")

    logger.info(f"\n1. Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info(f"\n2. Loading model from {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True
    )

    logger.info("\n3. Preparing model for QLoRA training...")
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    logger.info("\n4. Applying LoRA configuration...")
    lora_config = create_qlora_config()
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"\nTrainable parameters: {trainable_params:,}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

    logger.info(f"\n5. Loading dataset from {DATASET_FILE}...")
    dataset = load_and_prepare_data(DATASET_FILE)
    logger.info(f"Loaded {len(dataset)} training examples")

    logger.info("\n6. Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    logger.info("\n7. Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=False,
        save_steps=50,
        logging_steps=10,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",
        load_best_model_at_end=False,
        warmup_steps=10,
        optim="adamw_torch",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    logger.info("\n8. Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    logger.info("\n9. Starting training...")
    logger.info("This may take a while on CPU...\n")
    trainer.train()

    logger.info(f"\n10. Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {OUTPUT_DIR}")

    logger.info("\n11. Testing the fine-tuned model...")
    test_inference(model, tokenizer)


def test_inference(model, tokenizer):
    model.eval()

    test_prompts = [
        "<s>[INST] Suggest a recipe using eggs and onions\nIngredients: eggs, onions [/INST]",
        "<s>[INST] What can I make with tomatoes and pasta\nIngredients: tomatoes, pasta [/INST]",
    ]

    logger.info("\nTest Results:")

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=200, temperature=0.7, do_sample=True, top_p=0.9
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"\nPrompt: {prompt[:50]}...")
        logger.info(f"Response: {response[len(prompt) :]}")


if __name__ == "__main__":
    if not os.path.exists(DATASET_FILE):
        logger.info(f"Error: {DATASET_FILE} not found!")
        logger.info("Please run create_recipe_dataset.py first")
        exit(1)

    main()
