from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
import os

print(os.environ.get('OPENAI_API_KEY'))
# source ~/.zshrc

# loader = PyPDFLoader("./Atomic_Habits.pdf")

# pages = loader.load()

# print(len(pages))