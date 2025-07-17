import os
import requests
import speech_recognition as sr
from pydub import AudioSegment
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
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

# === Load BLIP image captioning model ===
print("[INFO] Loading BLIP image captioning model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("[INFO] BLIP model loaded.")

# === Generate Caption from Image ===
def generate_image_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.strip()

# === Language Tag (for voice only) ===
def get_language_tag(lang_code: str) -> str:
    return {
        "bn": "🔤 Detected Language: Bangla 🇧🇩",
        "en": "🔤 Detected Language: English 🇬🇧"
    }.get(lang_code, "🔤 Detected Language: Unknown ❓")

# === Split long messages to avoid Telegram's 4096-character limit ===
async def send_long_message(update: Update, text: str, max_length=4096):
    if len(text) <= max_length:
        await update.message.reply_text(text)
        return
    parts = []
    current_part = ""
    for line in text.split("\n"):
        if len(current_part) + len(line) + 1 <= max_length:
            current_part += line + "\n"
        else:
            parts.append(current_part.strip())
            current_part = line + "\n"
    if current_part:
        parts.append(current_part.strip())
    for part in parts:
        await update.message.reply_text(part)

# === /start Command ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hello! Send your legal question as text, voice, or image.")

# === Handle Text Message ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    try:
        res = requests.post(API_URL, json={"question": question})
        res.raise_for_status()  # Raise an exception for bad status codes
        response_data = res.json()
        print(f"API Response: {response_data}")  # Debug: Check API response
        answer = response_data.get("answer", "❌ Sorry, couldn't find an answer.")
        confidence = response_data.get("confidence", 0.0)
        await send_long_message(update, f"📜 Answer:\n{answer}\n\n⭐ Confidence: {confidence}/10")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

# === Handle Voice Message ===
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await update.message.voice.get_file()
    ogg_path = "voice.ogg"
    wav_path = "voice.wav"
    await file.download_to_drive(ogg_path)

    sound = AudioSegment.from_file(ogg_path)
    sound.export(wav_path, format="wav")

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)

    try:
        raw_text = recognizer.recognize_google(audio, language="en-US").lower()
    except:
        await update.message.reply_text("❌ Could not understand the voice message.")
        return

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

    await update.message.reply_text(f"🗣 Transcribed: {final_text}\n{get_language_tag(lang)}")

    try:
        res = requests.post(API_URL, json={"question": final_text})
        res.raise_for_status()  # Raise an exception for bad status codes
        response_data = res.json()
        print(f"API Response: {response_data}")  # Debug: Check API response
        answer = response_data.get("answer", "❌ Sorry, couldn't find an answer.")
        confidence = response_data.get("confidence", 0.0)
        await send_long_message(update, f"📜 Answer:\n{answer}\n\n⭐ Confidence: {confidence}/10")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

# === Handle Photo/Image Message ===
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]  # highest resolution
    image_path = "photo.jpg"
    await (await photo.get_file()).download_to_drive(image_path)

    try:
        # Generate basic caption
        caption = generate_image_caption(image_path)

        # Create detailed prompt with legal question
        detailed_prompt = (
            f"Describe in detail what is happening in the following image:\n{caption}\n\n"
            "Now, Based on this description, which law in Bangladesh is potentially being violated, and what type of punishment could apply under that law, including how many years of imprisonment or how much fine may be imposed?"
        )

        # Inform user what is being processed
        await send_long_message(update, f"🖼 Processing image and asking AI:\n{detailed_prompt}")

        # Send prompt to FastAPI LLM
        res = requests.post(API_URL, json={"question": detailed_prompt})
        res.raise_for_status()  # Raise an exception for bad status codes
        response_data = res.json()
        print(f"API Response: {response_data}")  # Debug: Check API response
        answer = response_data.get("answer", "❌ Sorry, couldn't find an answer.")
        confidence = response_data.get("confidence", 0.0)
        await send_long_message(update, f"📜 AI Answer:\n{answer}\n\n⭐ Confidence: {confidence}/10")
    except Exception as e:
        await update.message.reply_text(f"❌ Error processing image: {str(e)}")

# === Main function to run the bot ===
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    print("🤖 Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()