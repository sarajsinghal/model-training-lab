# Optional Hugging Face fine-tuning in Google Colab

This optional route shows how to adapt a small pretrained classifier to labeled text. It is not required for the local demo or baseline, and it is not training a foundation model from scratch. Colab resources and free-tier availability can change; never upload private, regulated, or unlicensed data to a third-party notebook.

1. Create a Colab notebook and select CPU first. A GPU is optional and never assumed.
2. Upload a CSV with `text` and `label` columns, or use `dataset.csv`.
3. Install compatible packages:

   ```python
   %pip install "transformers>=4.48,<5" "datasets>=3,<5" "evaluate>=0.4,<1" "accelerate>=1,<2" "scikit-learn>=1.5,<2"
   ```

4. Start with `distilbert/distilbert-base-uncased` or another small documented checkpoint. Downloading weights needs internet and may consume substantial RAM/disk. This is not the default for an 8 GB laptop.
5. Create train/validation/test partitions with a fixed seed. Tokenize with truncation and max length 128; use batch size 4 or 8.
6. Fine-tune only on training data, select settings on validation, and evaluate once on test. Record model revision, package versions, seed, dataset hash, settings and held-out metrics.
7. Export model and tokenizer using `save_pretrained`, plus a model card. Check licensing and privacy before sharing.

## Starter sketch

This outline requires `label_names` and separate Hugging Face Dataset objects. Trainer APIs evolve; pin versions and consult their documentation.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

checkpoint = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=len(label_names),
    id2label={i: name for i, name in enumerate(label_names)},
    label2id={name: i for i, name in enumerate(label_names)},
)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)

# train_ds = train_ds.map(tokenize, batched=True)
# val_ds = val_ds.map(tokenize, batched=True)
# test_ds = test_ds.map(tokenize, batched=True)
args = TrainingArguments(
    output_dir="./outputs", num_train_epochs=1,
    per_device_train_batch_size=4, per_device_eval_batch_size=4,
    learning_rate=2e-5, weight_decay=0.01,
    eval_strategy="epoch", save_strategy="no", report_to="none",
    seed=42, dataloader_num_workers=0,
)
# trainer = Trainer(model=model, args=args, train_dataset=train_ds,
#                   eval_dataset=val_ds, processing_class=tokenizer)
# trainer.train()
# final_metrics = trainer.evaluate(test_ds)
# trainer.save_model("./fine_tuned_model")
# tokenizer.save_pretrained("./fine_tuned_model")
```

Weights download only in this optional notebook. The local app uses no paid APIs or API keys.

