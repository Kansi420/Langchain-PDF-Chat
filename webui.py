import os
os.environ["OPENAI_API_KEY"] = ""

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
embeddings = OpenAIEmbeddings()
docsearch = FAISS.load_local("faiss_index", embeddings)

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

chain = load_qa_chain(OpenAI(), chain_type="stuff")

# query = "what are the fundaments according to this book"
# docs = docsearch.similarity_search(query)
# print(chain.run(input_documents=docs, question=query))

import gradio as gr

def greet(name):
    docs = docsearch.similarity_search(name)
    return chain.run(input_documents=docs, question=name)


myServer = gr.Interface(fn=greet, inputs="text", outputs="text")

myServer.launch(server_name="0.0.0.0")