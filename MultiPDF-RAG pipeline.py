from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

def load_and_process_pdf(pdf_files):
    for pdf_file in pdf_files:
        # Read text from PDF
        pdfreader = PdfReader(pdf_file)
        raw_text = ''
        for page in pdfreader.pages:
            content = page.extract_text()
            if content:
                raw_text += content
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    return texts

splits = load_and_process_pdf([".venv\Erioluwa.pdf", ".venv/Dialectic.pdf", ".venv\Collective_unconscious.pdf", ".venv\Socratic_method.pdf"])

def initialize_vectorstore(splits):
    embeddings = get_embedding_function()
    return FAISS.from_texts(splits, embeddings)

vectorstore = initialize_vectorstore(splits)

chain = load_qa_chain(llm=Ollama(model="mistral"), chain_type="stuff")
query = "What is a dialectic"
docs = vectorstore.similarity_search(query)
input_data = {
    'input_documents': docs,  # Provide the retrieved documents
    'input': query , # Provide the query text
    'question': query,
}
print(chain.invoke(input_data)["output_text"])