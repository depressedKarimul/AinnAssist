import os
import requests
import speech_recognition as sr
from pydub import AudioSegment
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

# === Set FFmpeg path for pydub ===
AudioSegment.converter = "C:\\ffmpeg\\bin\\ffmpeg.exe"
AudioSegment.ffprobe = "C:\\ffmpeg\\bin\\ffprobe.exe"

BOT_TOKEN = ""
API_URL = "http://localhost:8000/ask"

# === Start Command ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hello! Send a legal question as text or voice message.")

# === Handle Text Messages ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    try:
        res = requests.post(API_URL, json={"question": user_question})
        answer = res.json().get("answer", "❌ Sorry, couldn't find an answer.")
        await update.message.reply_text(f"📜 Answer:\n{answer}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")
# === Handle Voice Messages ===
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await update.message.voice.get_file()
    file_path = "voice.ogg"
    await file.download_to_drive(file_path)

    # Convert OGG to WAV
    wav_path = "voice.wav"
    sound = AudioSegment.from_file(file_path)
    sound.export(wav_path, format="wav")

    # Speech Recognition
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)

    try:
        transcribed_text = recognizer.recognize_google(audio).strip()

        # Ensure the question ends with a '?'
        if not transcribed_text.endswith("?"):
            transcribed_text += "?"

        await update.message.reply_text(f"🗣 Transcribed: {transcribed_text}")

        # Send to API
        res = requests.post(API_URL, json={"question": transcribed_text})
        answer = res.json().get("answer", "❌ Sorry, couldn't find an answer.")
        await update.message.reply_text(f"📜 Answer:\n{answer}")
    except Exception as e:
        await update.message.reply_text(f"❌ Voice processing error: {str(e)}")


# === Bot Main Setup ===
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    print("🤖 Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
