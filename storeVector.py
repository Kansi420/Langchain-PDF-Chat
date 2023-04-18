import os
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI



os.environ["OPENAI_API_KEY"] = "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"

loader = PyPDFLoader("./Atomic_Habits.pdf")
pages = loader.load()

raw_text = ''
for i, page in enumerate(pages):
    text = page.page_content
    if text:
        raw_text += text

print(raw_text[:50])

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

text_splitter = CharacterTextSplitter(
    separator= "\n",
    chunk_size=500,
    chunk_overlap=50,
    )

texts = text_splitter.split_text(raw_text)

embeddings = OpenAIEmbeddings()

print(len(texts))

print(texts[50])

from langchain.vectorstores import FAISS
docsearch = FAISS.from_texts(texts, embeddings)

docsearch.save_local("faiss_index")

# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI

# chain = load_qa_chain(OpenAI(), chain_type="stuff")

# query = "what are the fundaments according to this book"
# docs = docsearch.similarity_search(query)
# print(chain.run(input_documents=docs, question=query))

# import gradio as gr

# def greet(name):
#     return "Hello " + name + "!!"


# myServer = gr.Interface(fn=greet, inputs="text", outputs="text")

# myServer.launch(server_name="0.0.0.0")