"""
Medical QA Assistant Application

This module provides a Gradio-based web interface for a medical question-answering assistant.
It allows users to ask questions about patient health information and retrieves relevant
context from a knowledge base using vector search and language models.

The application consists of:
- A chat interface for user interaction
- Context retrieval and display from medical documents
- Integration with a custom chat module for answer generation
"""

import gradio as gr
from dotenv import load_dotenv
from chat import answer_question


load_dotenv(override=True)


def format_context(context):
    """
    Format the retrieved context documents for display in the Gradio interface.

    This function takes a list of document objects (likely from vector search results)
    and formats them into HTML for display in the context panel. Each document
    includes its source metadata and page content.

    Args:
        context (list): List of document objects with metadata and page_content attributes.

    Returns:
        str: Formatted HTML string containing the context information.
    """
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in context:
        result += f"<span style='color: #ff7800;'>Source: {doc.metadata['source']}</span>\n\n"
        result += doc.page_content + "\n\n"
    return result


def chat(history):
    """
    Process a chat message and generate a response with context.

    This function is called when the user submits a message in the chat interface.
    It extracts the last user message from the chat history, calls the answer_question
    function to get an AI-generated response and relevant context, then appends
    the assistant's response to the chat history.

    Args:
        history (list): List of chat messages in Gradio's format, where each message
                       is a dict with 'role' and 'content' keys.

    Returns:
        tuple: (updated_history, formatted_context) where updated_history includes
               the new assistant message, and formatted_context is HTML for display.
    """
    last_message = history[-1]["content"][0]['text']
    prior = history[:-1]
    answer, context = answer_question(last_message, prior)
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)


def main():
    """
    Set up and launch the Gradio web interface for the medical QA assistant.

    This function creates the UI components including the chatbot, message input,
    and context display panel. It defines the event handlers for user interactions
    and launches the application in the browser.
    """
    def put_message_in_chatbot(message, history):
        """
        Helper function to add user message to chat history.

        Args:
            message (str): The user's input message.
            history (list): Current chat history.

        Returns:
            tuple: ("", updated_history) to clear the input and update history.
        """
        return "", history + [{"role": "user", "content": message}]

    # Use a soft theme with specified fonts for better readability
    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    # Create the main Gradio Blocks interface
    with gr.Blocks(title="Patient Health Information Retrieval Expert Assistant") as ui:
        # Main title and description
        gr.Markdown(
            "# 🏢 Patient Health Information Retrieval Expert Assistant\nAsk me anything about !")

        # Layout with two columns: chat on left, context on right
        with gr.Row():
            with gr.Column(scale=1):
                # Chatbot component for conversation display
                chatbot = gr.Chatbot(
                    label="💬 Conversation", height=600
                )
                # Text input for user questions
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about patient health Information...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                # Markdown component to display retrieved context
                context_markdown = gr.Markdown(
                    label="📚 Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        # Event handling: when message is submitted, add to chat then process
        message.submit(
            put_message_in_chatbot, inputs=[
                message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown])

    # Launch the interface in the browser
    ui.launch(theme=theme, inbrowser=True)


# Standard Python entry point: run the main function when script is executed directly
if __name__ == "__main__":
    main()
