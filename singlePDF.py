from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms.ollama import Ollama
from typing_extensions import Concatenate
from langchain_community.embeddings.ollama import OllamaEmbeddings


pdfreader = PdfReader('insert pdf path')
def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

#STEP 1
# read text from pdf
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

#STEP 2: EMBED DOCUMENT using FAISS
embeddings = get_embedding_function()
document_search = FAISS.from_texts(texts, embeddings)

#STEP 3: USE QA CHAIN TO PULL TEXT RELATED TO QUERY
chain = load_qa_chain(llm=Ollama(model="mistral"), chain_type="stuff")
query = "What is the method of elenchus"
docs = document_search.similarity_search(query)
input_data = {
    'input_documents': docs,  # Provide the retrieved documents
    'input': query , # Provide the query text
    'question': query,
}
print(chain.invoke(input_data)["output_text"])
