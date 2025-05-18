from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# مسیر فایل دیتاست
dataset_path = "dataset.txt"

# ساخت توکنایزر BPE
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# آموزش توکنایزر روی دیتاست
trainer = trainers.BpeTrainer(
    vocab_size=1000,
    show_progress=True,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

tokenizer.train([dataset_path], trainer)

# ذخیره توکنایزر
tokenizer.save("tinygpt_tokenizer.json")
print("✅ توکنایزر با موفقیت ساخته شد.")
