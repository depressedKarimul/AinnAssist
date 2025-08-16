
import os
import requests
import speech_recognition as sr
from pydub import AudioSegment
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
from telegram import Update 
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters
)
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langdetect import detect

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

# === Language Tag ===
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
        # Detect language
        lang = detect(question)
        question_to_api = question

        # If Bangla, translate to English
        if lang == "bn":
            question_to_api = GoogleTranslator(source='bn', target='en').translate(question)
            await update.message.reply_text(f"🗣 Transcribed: {question}\n{get_language_tag(lang)}")

        # Send to FastAPI
        res = requests.post(API_URL, json={"question": question_to_api})
        res.raise_for_status()
        response_data = res.json()
        answer = response_data.get("answer", "❌ Sorry, couldn't find an answer.")
        confidence = response_data.get("confidence", 5.0)

        # If input was Bangla, translate answer back to Bangla
        if lang == "bn":
            answer = GoogleTranslator(source='en', target='bn').translate(answer)
        
        await send_long_message(update, f"📜 Answer:\n{answer}\n\n⭐ Confidence: {confidence}/5")
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
        # Translate to English if Bangla
        question_to_api = final_text
        if lang == "bn":
            question_to_api = GoogleTranslator(source='bn', target='en').translate(final_text)

        # Send to FastAPI
        res = requests.post(API_URL, json={"question": question_to_api})
        res.raise_for_status()
        response_data = res.json()
        answer = response_data.get("answer", "❌ Sorry, couldn't find an answer.")
        confidence = response_data.get("confidence", 5.0)

        # Translate answer to Bangla if input was Bangla
        if lang == "bn":
            answer = GoogleTranslator(source='en', target='bn').translate(answer)

        # Convert answer to voice
        tts = gTTS(answer, lang=lang)
        tts.save("response.mp3")
        sound = AudioSegment.from_mp3("response.mp3")
        sound.export("response.ogg", format="ogg", codec="libopus")

        # Send as voice
        with open("response.ogg", "rb") as voice_file:
            await update.message.reply_voice(voice=voice_file, caption=f"⭐ Confidence: {confidence}/5")

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")
    finally:
        for f in ["voice.ogg", "voice.wav", "response.mp3", "response.ogg"]:
            if os.path.exists(f):
                os.remove(f)

# === Handle Photo/Image Message ===
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    image_path = "photo.jpg"
    await (await photo.get_file()).download_to_drive(image_path)

    try:
        caption = generate_image_caption(image_path)
        detailed_prompt = (
            f"Describe in detail what is happening in the following image:\n{caption}\n\n"
            "Now, Based on this description, which law in Bangladesh is potentially being violated, and what type of punishment could apply under that law, including how many years of imprisonment or how much fine may be imposed?"
        )
        await send_long_message(update, f"🖼 Processing image and asking AI:\n{detailed_prompt}")
        res = requests.post(API_URL, json={"question": detailed_prompt})
        res.raise_for_status()
        response_data = res.json()
        answer = response_data.get("answer", "❌ Sorry, couldn't find an answer.")
        confidence = response_data.get("confidence", 5.0)
        await send_long_message(update, f"📜 AI Answer:\n{answer}\n\n⭐ Confidence: {confidence}/5")
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
	
	



