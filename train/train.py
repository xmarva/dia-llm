import os

import evaluate
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "EleutherAI/pythia-160m"
DATASET_NAME = "allenai/dialoglue"
DATASET_CONFIG = "wow"
OUTPUT_DIR = "./output"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
BATCH_SIZE = 8
GRAD_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3


def prepare_dataset(tokenizer):
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    def format_dialog(example):
        dialog = []
        for i in range(len(example["user_utterance"])):
            dialog.append(f"Human: {example['user_utterance'][i]}")
            if i < len(example["system_response"]):
                dialog.append(f"Assistant: {example['system_response'][i]}")

        return {"text": "\n".join(dialog)}

    tokenized_datasets = {}
    for split in dataset.keys():
        formatted_dataset = dataset[split].map(
            format_dialog, remove_columns=dataset[split].column_names
        )

        def tokenize_function(examples):
            return tokenizer(
                examples["text"], truncation=True, max_length=512, padding="max_length"
            )

        tokenized_datasets[split] = formatted_dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

    return tokenized_datasets


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    tokenized_datasets = prepare_dataset(tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
        report_to="tensorboard",
        logging_steps=100,
    )

    metric = evaluate.load("perplexity")

    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids

        perplexity = metric.compute(predictions=logits, references=labels)

        return {"perplexity": perplexity["perplexity"]}

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))

    print("Training completed. Model saved to", os.path.join(OUTPUT_DIR, "final_model"))

    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    return os.path.join(OUTPUT_DIR, "final_model")


if __name__ == "__main__":
    train()
