import torch
from tinygpt_model import TinyGPT
from tokenizers import Tokenizer
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# تنظیمات پایه
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 32
MAX_GEN = 50

# بارگذاری مدل و توکنایزر
tokenizer = Tokenizer.from_file("tinygpt_tokenizer.json")
vocab_size = tokenizer.get_vocab_size()

model = TinyGPT(vocab_size)
model.load_state_dict(torch.load("tinygpt_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# تابع تولید پاسخ
def generate(prompt, max_gen=MAX_GEN):
    ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    for _ in range(max_gen):
        if input_ids.size(1) > SEQ_LEN:
            input_ids = input_ids[:, -SEQ_LEN:]

        with torch.no_grad():
            logits = model(input_ids)
            next_id = torch.argmax(logits[0, -1]).item()

        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=DEVICE)], dim=1)

    return tokenizer.decode(input_ids[0].tolist())

# هندلر اصلی پیام
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_msg = update.message.text
    reply = generate(user_msg)
    await update.message.reply_text(reply)

# راه‌اندازی بات
if __name__ == "__main__":
    TOKEN = "7780785188:AAGT1smLb4caAejpPJIIMqLShC6aFZ0_zow"
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle))
    print("🤖 بات روشنه و منتظر پیامه...")
    app.run_polling()
