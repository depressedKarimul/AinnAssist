# Welcome to AinnAssist 👋


## 🔧 Installation

To get started, make sure you have [Pipenv](https://pipenv.pypa.io/en/latest/) installed. Then run the following command in your terminal:

```bash
pipenv install langchain langchain_community langchain_ollama langchain_core langchain_groq faiss-cpu pdfplumber

```
After setting up the Pipenv environment, also install the pypdf library:

```bash
pip install pypdf

```


🚀 Run the Script

To initialize memory for the LLM, run:

```bash

pipenv run python create_memory_for_llm.py


```
activate the virtual environment first:

```bash

pipenv shell


```

Launch the Streamlit app:

Install Streamlit (if not already installed):

```bash
pip install streamlit
```
Run the application:

```bash

streamlit run app.py

```

Install FastAPI & Uvicorn

```bash
pip install fastapi uvicorn python-multipart

```

Run your API

```bash
uvicorn app:app --reload

```

## Telegram Bot with Voice Message Support


## 🛠️ Requirements

### 1. Install FFmpeg

#### Windows:

- Download FFmpeg from the official site:  
  [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

- Extract the folder and add its `bin` path (e.g., `C:\ffmpeg\bin`) to your system **Environment Variables > PATH**.

To verify FFmpeg is installed:

```bash

ffmpeg -version

```

📦 Install Required Python Packages
Go to your project folder and run:

```bash

pip install python-telegram-bot==20.3 pydub SpeechRecognition requests

```

🚀 Running the Bot
After setting up everything, run the bot using:

```bash

python bot.py


```