import os
import requests
import speech_recognition as sr
from pydub import AudioSegment
from langdetect import detect
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

# === Set FFmpeg path for pydub ===
AudioSegment.converter = "C:\\ffmpeg\\bin\\ffmpeg.exe"
AudioSegment.ffprobe = "C:\\ffmpeg\\bin\\ffprobe.exe"

BOT_TOKEN = ""
API_URL = "http://localhost:8000/ask"

# === Language Detection Utility ===
def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "unknown"

def get_language_tag(lang_code: str) -> str:
    return {
        "bn": "🔤 Detected Language: Bangla 🇧🇩",
        "en": "🔤 Detected Language: English 🇬🇧"
    }.get(lang_code, "🔤 Detected Language: Unknown ❓")

# === Bot Start Command ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hello! Send a legal question as text or voice message.")

# === Text Message Handler ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    lang = detect_language(user_question)
    lang_tag = get_language_tag(lang)

    try:
        res = requests.post(API_URL, json={"question": user_question})
        answer = res.json().get("answer", "❌ Sorry, couldn't find an answer.")
        await update.message.reply_text(f"{lang_tag}\n\n📜 Answer:\n{answer}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

# === Voice Message Handler ===
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

    # Try both languages
    try:
        bn_text = recognizer.recognize_google(audio, language="bn-BD")
    except:
        bn_text = ""
    try:
        en_text = recognizer.recognize_google(audio, language="en-US")
    except:
        en_text = ""

    bn_detected = detect_language(bn_text) if bn_text else "unknown"
    en_detected = detect_language(en_text) if en_text else "unknown"

    # Choose the best transcription based on langdetect and script
    if en_detected == "en" and not bn_detected == "bn":
        final_text = en_text
        lang = "en"
    elif bn_detected == "bn" and not en_detected == "en":
        final_text = bn_text
        lang = "bn"
    elif en_text and bn_text:
        # fallback: more ascii = likely English
        en_score = sum(c.isascii() for c in en_text)
        bn_score = sum(c.isascii() for c in bn_text)
        if en_score >= bn_score:
            final_text = en_text
            lang = "en"
        else:
            final_text = bn_text
            lang = "bn"
    else:
        await update.message.reply_text("❌ Could not understand the voice message.")
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


# === Main ===
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    print("🤖 Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
