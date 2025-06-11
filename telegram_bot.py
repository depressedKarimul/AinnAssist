import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

BOT_TOKEN = ""
API_URL = "http://localhost:8000/ask"  # Your FastAPI endpoint

# Command to start the bot
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hello! Ask me any legal question based on Bangladeshi law or the UDHR.")

# Message handler
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    try:
        response = requests.post(API_URL, json={"question": user_question})
        answer = response.json().get("answer", "Sorry, I couldn't find an answer.")
        await update.message.reply_text(f"📜 Answer:\n{answer}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

# Main bot setup
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("🤖 Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
