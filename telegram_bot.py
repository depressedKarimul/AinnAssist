import os
import requests
import speech_recognition as sr
from pydub import AudioSegment
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters
)
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
API_URL = os.getenv("API_URL")
FFMPEG_PATH = os.getenv("FFMPEG_PATH")
FFPROBE_PATH = os.getenv("FFPROBE_PATH")

# === Set FFmpeg path for pydub ===
AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffprobe = FFPROBE_PATH

# === Language Tag (for voice only) ===
def get_language_tag(lang_code: str) -> str:
    return {
        "bn": "🔤 Detected Language: Bangla 🇧🇩",
        "en": "🔤 Detected Language: English 🇬🇧"
    }.get(lang_code, "🔤 Detected Language: Unknown ❓")

# === /start Command ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hello! Send a legal question as text or voice message.")

# === Handle Text Message ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text

    try:
        res = requests.post(API_URL, json={"question": user_question})
        answer = res.json().get("answer", "❌ Sorry, couldn't find an answer.")
        await update.message.reply_text(f"📜 Answer:\n{answer}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

# === Handle Voice Message ===
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await update.message.voice.get_file()
    file_path = "voice.ogg"
    await file.download_to_drive(file_path)

    # Convert OGG to WAV
    wav_path = "voice.wav"
    sound = AudioSegment.from_file(file_path)
    sound.export(wav_path, format="wav")

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)

    try:
        raw_text = recognizer.recognize_google(audio, language="en-US").lower()
    except:
        await update.message.reply_text("❌ Could not understand the voice message.")
        return

    print("Initial voice text:", raw_text)

    if raw_text.startswith("bangla") or raw_text.startswith("বাংলা"):
        try:
            bn_text = recognizer.recognize_google(audio, language="bn-BD")
            final_text = bn_text.lstrip("বাংলা").lstrip("Bangla").strip()
            lang = "bn"
        except:
            await update.message.reply_text("❌ Could not transcribe in Bangla.")
            return
    else:
        try:
            final_text = recognizer.recognize_google(audio, language="en-US")
            lang = "en"
        except:
            await update.message.reply_text("❌ Could not transcribe in English.")
            return

    if not final_text.endswith("?"):
        final_text += "?"

    lang_tag = get_language_tag(lang)
    await update.message.reply_text(f"🗣 Transcribed: {final_text}\n{lang_tag}")

    try:
        res = requests.post(API_URL, json={"question": final_text})
        answer = res.json().get("answer", "❌ Sorry, couldn't find an answer.")
        await update.message.reply_text(f"📜 Answer:\n{answer}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

# === Main Entry ===
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    print("🤖 Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
