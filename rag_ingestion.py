import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def load_document(path: str):
    print("Loading The data from document....")
    loader = TextLoader(file_path=path, encoding="UTF-8")
    docs = loader.load()

    print("Document Loaded Successfully...")
    return docs


def split_dcoument(docs):
    print("Text Splitting is started...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)
    print("Text Splitting is completed...")
    return texts


if __name__ == "__main__":
    print("ingesting data...")
    print(os.environ.get("PINECONE_API_KEY"))
    documents = load_document("burj_khalifa.txt")
    texts = split_dcoument(documents)
    embedding = OpenAIEmbeddings(
        api_key=os.environ.get("OPENAI_API_KEY"), model="text-embedding-3-small"
    )
    # pc=Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    # index=pc.Index(os.environ.get(""))
    PineconeVectorStore.from_documents(
        texts, embedding, index_name=os.environ.get("PINECONE_INDEX_NAME")
    )
    print("ingestion is completed...")
