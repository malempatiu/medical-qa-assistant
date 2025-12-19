import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from a .env file, overriding any existing environment variables.
# This ensures that API keys and other configurations are set from the file.
load_dotenv(override=True)

# DB_NAME defines the path to the directory where the vector database will be stored.
# This directory holds the persisted embeddings and metadata for the Chroma vector store.
# The path is constructed relative to the script's directory to ensure portability.
# Chroma uses this directory to save and load the vector database for efficient retrieval.
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

# KNOWLEDGE_BASE specifies the path to the directory containing the source documents
# that will be ingested into the knowledge base. These documents are loaded, split into chunks,
# and embedded to create the searchable vector database. The path is relative to the script's
# directory, allowing the knowledge base to be organized alongside the ingestion script.
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

# Specify the OpenAI model to use for potential future use in generation or other tasks.
# Currently, this variable is defined but not used in this script.
MODEL = "gpt-4.1-nano"

# Initialize the OpenAI embeddings model for converting text into vector representations.
# The "text-embedding-3-large" model is used for high-quality embeddings suitable for semantic search.
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def fetch_documents():
    """
    Load documents from the knowledge base directory.

    This function uses PyPDFDirectoryLoader to load all PDF files from the KNOWLEDGE_BASE directory.
    It determines the document type from the file extension in the document's metadata 'source' field.
    Returns a list of loaded documents ready for further processing.
    """
    loader = PyPDFDirectoryLoader(KNOWLEDGE_BASE)
    documents = []
    docs = loader.load()
    for doc in docs:
        # Determine document type from file extension
        source_path = doc.metadata.get('source', '')
        if source_path:
            doc_type = Path(source_path).suffix[1:].lower()  # e.g., 'pdf'
        else:
            doc_type = 'unknown'
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)
    return documents


def create_chunks(documents):
    """
    Split the loaded documents into smaller chunks for embedding.

    Uses RecursiveCharacterTextSplitter to divide documents into chunks of 500 characters
    with a 200-character overlap to maintain context across chunks.
    This helps in creating more coherent and searchable text segments.
    Returns a list of document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_embeddings(chunks):
    """
    Create and store embeddings for the document chunks in the vector database.

    If a vector database already exists at DB_NAME, it deletes the existing collection
    to ensure a fresh start. Then, it creates a new Chroma vector store from the chunks
    using the specified embeddings model and persists it to the DB_NAME directory.
    Finally, it prints statistics about the number of vectors and their dimensions.
    Returns the created vector store object.
    """
    if os.path.exists(DB_NAME):
        Chroma(
            persist_directory=DB_NAME,
            embedding_function=embeddings
        ).delete_collection()

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    collection = vector_store._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])[
        "embeddings"][0]
    dimensions = len(sample_embedding)
    print(
        f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vector_store


if __name__ == "__main__":
    # Main execution block to run the ingestion process when the script is executed directly.
    # This block orchestrates the entire pipeline: fetching documents, creating chunks,
    # generating embeddings, and storing them in the vector database.
    # Prints a completion message once the process is done.
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
