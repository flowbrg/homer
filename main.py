# gradio_app.py
import gradio as gr
import time
from typing import List, Tuple, Optional
from datetime import datetime

# Import your existing modules
from src.core.application import Application
from src.core.configuration import Configuration
from src.core import database
from src.resources.utils import load_config
from langchain_core.messages import AIMessage, HumanMessage

# Global variables
app_instance = None
current_thread_id = None

def initialize_app():
    """Initialize the application."""
    global app_instance
    if app_instance is None:
        config = load_config()
        app_instance = Application(config)

def get_threads() -> List[Tuple[str, int]]:
    """Get all available threads."""
    threads = database.get_all_threads()
    return [(f"💬 {name}", thread_id) for thread_id, name in threads]

def create_thread(thread_name: str) -> Tuple[gr.Dropdown, str]:
    """Create a new thread."""
    if not thread_name.strip():
        return gr.Dropdown(choices=get_threads()), "❌ Le nom ne peut pas être vide"
    
    existing_threads = database.get_all_threads()
    new_id = (existing_threads[-1][0] + 1) if existing_threads else 1
    
    database.new_thread(thread_id=new_id, thread_name=thread_name.strip())
    
    # Update dropdown choices
    updated_choices = get_threads()
    return (
        gr.Dropdown(choices=updated_choices, value=f"💬 {thread_name.strip()}"),
        f"✅ Thread '{thread_name}' créé avec succès!"
    )

def delete_current_thread(selected_thread: str) -> Tuple[gr.Dropdown, gr.Chatbot, str]:
    """Delete the currently selected thread."""
    global current_thread_id
    
    if not selected_thread or current_thread_id is None:
        return gr.Dropdown(), gr.Chatbot([]), "❌ Aucun thread sélectionné"
    
    database.delete_thread(current_thread_id)
    current_thread_id = None
    
    updated_choices = get_threads()
    return (
        gr.Dropdown(choices=updated_choices, value=None),
        gr.Chatbot([]),
        "✅ Thread supprimé avec succès!"
    )

def load_thread_messages(selected_thread: str) -> gr.Chatbot:
    """Load messages for the selected thread."""
    global current_thread_id
    
    if not selected_thread:
        current_thread_id = None
        return gr.Chatbot([])
    
    # Extract thread ID from selection
    threads = database.get_all_threads()
    thread_name = selected_thread.replace("💬 ", "")
    
    for thread_id, name in threads:
        if name == thread_name:
            current_thread_id = thread_id
            break
    else:
        current_thread_id = None
        return gr.Chatbot([])
    
    # Load messages
    messages = app_instance.get_messages(current_thread_id)
    chat_history = []
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            chat_history.append([msg.content, None])
        elif isinstance(msg, AIMessage):
            if chat_history and chat_history[-1][1] is None:
                chat_history[-1][1] = msg.content
            else:
                chat_history.append([None, msg.content])
    
    return gr.Chatbot(chat_history)

def respond_to_message(message: str, chat_history: List[List[str]]) -> Tuple[gr.Chatbot, str]:
    """Process user message and generate AI response."""
    global current_thread_id
    
    if not message.strip():
        return gr.Chatbot(chat_history), ""
    
    if current_thread_id is None:
        return gr.Chatbot(chat_history), "❌ Veuillez sélectionner un thread d'abord"
    
    # Add user message to chat
    chat_history.append([message, None])
    
    try:
        # Stream AI response
        ai_response = ""
        for token in app_instance.stream_retrieval_graph(query=message, thread_id=current_thread_id):
            ai_response += token
            # Update the last message with the current response
            chat_history[-1][1] = ai_response
            yield gr.Chatbot(chat_history), ""
            time.sleep(0.02)  # Small delay for streaming effect
        
        # Final update
        chat_history[-1][1] = ai_response
        yield gr.Chatbot(chat_history), ""
        
    except Exception as e:
        chat_history[-1][1] = f"❌ Erreur: {str(e)}"
        yield gr.Chatbot(chat_history), ""

def create_interface():
    """Create the Gradio interface."""
    initialize_app()
    
    # Custom CSS
    css = """
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .chatbot {
        height: 500px;
    }
    
    .message {
        padding: 10px;
        margin: 5px;
        border-radius: 10px;
    }
    
    .user-message {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        margin-left: 20%;
    }
    
    .bot-message {
        background: #f8fafc;
        color: #1f2937;
        margin-right: 20%;
        border: 1px solid #e2e8f0;
    }
    """
    
    with gr.Blocks(css=css, title="🤖 AI Chat Assistant", theme=gr.themes.Soft()) as interface:
        
        # Header
        gr.Markdown("""
        # 🤖 AI Chat Assistant
        
        Bienvenue dans votre assistant IA personnel ! Sélectionnez une conversation ou créez-en une nouvelle pour commencer.
        """)
        
        with gr.Row():
            # Left column - Thread management
            with gr.Column(scale=1):
                gr.Markdown("### 💬 Gestion des conversations")
                
                # Thread selection
                thread_dropdown = gr.Dropdown(
                    choices=get_threads(),
                    label="Sélectionner une conversation",
                    value=None,
                    interactive=True
                )
                
                # New thread creation
                with gr.Group():
                    gr.Markdown("**Nouvelle conversation**")
                    new_thread_name = gr.Textbox(
                        label="Nom de la conversation",
                        placeholder="Ex: Aide Python, Questions générales..."
                    )
                    create_btn = gr.Button("➕ Créer", variant="primary")
                
                # Thread management
                delete_btn = gr.Button("🗑️ Supprimer la conversation", variant="stop")
                
                # Status messages
                status_msg = gr.Textbox(
                    label="Status",
                    value="Prêt",
                    interactive=False,
                    lines=2
                )
            
            # Right column - Chat interface
            with gr.Column(scale=2):
                gr.Markdown("### 💭 Conversation")
                
                # Chat display
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                    show_label=False,
                    avatar_images=("👤", "🤖"),
                    bubble_full_width=False
                )
                
                # Message input
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Votre message",
                        placeholder="Tapez votre message ici...",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("📤 Envoyer", variant="primary", scale=1)
        
        # Event handlers
        
        # Create new thread
        create_btn.click(
            fn=create_thread,
            inputs=[new_thread_name],
            outputs=[thread_dropdown, status_msg]
        ).then(
            fn=lambda: "",  # Clear the input
            outputs=[new_thread_name]
        )
        
        # Delete thread
        delete_btn.click(
            fn=delete_current_thread,
            inputs=[thread_dropdown],
            outputs=[thread_dropdown, chatbot, status_msg]
        )
        
        # Load thread messages when selection changes
        thread_dropdown.change(
            fn=load_thread_messages,
            inputs=[thread_dropdown],
            outputs=[chatbot]
        )
        
        # Send message
        send_btn.click(
            fn=respond_to_message,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        # Send message on Enter
        msg_input.submit(
            fn=respond_to_message,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        # Footer
        gr.Markdown("""
        ---
        💡 **Astuce:** Utilisez la barre latérale pour gérer vos conversations. Vos messages sont automatiquement sauvegardés.
        """)
    
    return interface

# Launch the application
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file if needed

    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True if you want to share publicly
        debug=True,
        show_api=False
    )