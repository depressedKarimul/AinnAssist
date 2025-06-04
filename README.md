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