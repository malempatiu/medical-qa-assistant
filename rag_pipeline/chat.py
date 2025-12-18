from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

from dotenv import load_dotenv


# Load environment variables from a .env file, overriding any existing ones.
# This ensures API keys and configurations are properly set.
load_dotenv(override=True)

# Specify the OpenAI model to use for generating responses.
# This model is used for conversational AI in the chat system.
MODEL = "gpt-4.1-nano"

# Define the path to the vector database directory.
# This directory contains the persisted Chroma vector store with document embeddings.
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

# Initialize the OpenAI embeddings model for converting text into vectors.
# Used for semantic search and retrieval from the vector store.
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Set the number of top documents to retrieve for context.
# Higher values provide more context but may increase response time.
RETRIEVAL_K = 10

# Define the system prompt for the AI assistant.
# This prompt sets the role and behavior of the assistant, emphasizing health information retrieval.
# It includes a placeholder for context that will be filled with retrieved documents.
SYSTEM_PROMPT = """
You are a specialized health information retrieval assistant designed to support nurses, physicians, and other hospital staff by providing accurate, 
evidence-based answers from hospital documentation, policies, clinical guidelines, and medical knowledge bases. 
Your primary function is to help clinical staff quickly access relevant information to support their decision-making and daily workflows.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

# Initialize the Chroma vector store from the persisted directory.
# This loads the pre-computed embeddings and allows for similarity searches.
vector_store = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)

# Create a retriever object from the vector store.
# The retriever is configured to return the top K similar documents for a query.
retriever = vector_store.as_retriever()

# Initialize the ChatOpenAI language model with specified temperature and model.
# Temperature 0 ensures deterministic responses for consistency in medical contexts.
llm = ChatOpenAI(temperature=0, model_name=MODEL)


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a given question.

    This function uses the retriever to find the top RETRIEVAL_K documents
    most similar to the input question. These documents provide context
    for generating accurate answers.

    Args:
        question (str): The user's question.

    Returns:
        list[Document]: A list of relevant documents from the vector store.
    """
    return retriever.invoke(question, k=RETRIEVAL_K)


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine the conversation history with the current question.

    To provide better context for retrieval, this function concatenates
    all previous user messages from the chat history with the current question.
    This helps in retrieving information that considers the full conversation.

    Args:
        question (str): The current user question.
        history (list[dict]): List of previous messages in the conversation.

    Returns:
        str: A combined string of historical user messages and the current question.
    """
    prior = "\n".join(message["content"][0]['text']
                      for message in history if message["role"] == "user")
    return prior + "\n" + question


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Generate an answer to the question using Retrieval-Augmented Generation (RAG).

    This function orchestrates the RAG process: it combines the question with history,
    retrieves relevant context documents, formats them into a system prompt,
    and uses the language model to generate a response. It also returns the
    retrieved documents for transparency or further processing.

    Args:
        question (str): The user's current question.
        history (list[dict]): List of previous messages in the conversation.

    Returns:
        tuple[str, list[Document]]: A tuple containing the AI's response and the list of context documents used.
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs
