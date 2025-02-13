import streamlit as st
from typing import List
from groq import Groq
from pinecone import Pinecone
from openai import AzureOpenAI
import logging
import time

# Page config
st.set_page_config(
    page_title="Glacien GST Assistant",
    page_icon="🤖",
    layout="centered"
)

# # Custom CSS to reduce spacing and make chat more compact
# st.markdown('''
# <style>
# .stApp {
#     max-width: 1200px;
#     margin: 0 auto;
# }

# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}

# /* Reduce spacing in main containers */
# .block-container {
#     padding-top: 2rem;
#     padding-bottom: 0rem;
#     padding-left: 2rem;
#     padding-right: 2rem;
# }

# /* Adjust message container spacing */
# .stChatMessage {
#     padding: 0.5rem 0 !important;
#     margin: 0.5rem 0 !important;
# }

# .stChatMessage > div {
#     padding: 0.5rem 1rem !important;
#     border-radius: 0.5rem;
# }

# /* Reduce markdown spacing */
# .stMarkdown {
#     margin-bottom: 0 !important;
# }

# .stMarkdown p {
#     margin-bottom: 0;
# }

# /* Compact chat input */
# .stChatInput {
#     padding-bottom: 0.5rem;
#     padding-top: 0.5rem;
# }

# /* Adjust spacing between messages */
# .element-container {
#     margin-bottom: 0.2rem;
# }

# /* Message backgrounds */
# .stChatMessage.user > div {
#     background-color: #f7f7f8 !important;
# }

# .stChatMessage.assistant > div {
#     background-color: #ffffff !important;
# }
# </style>
# ''', unsafe_allow_html=True)

    
#     .element-container {
#         margin-bottom: 0.5rem;
#     }
    
#     .stMarkdown p {
#         margin-bottom: 0px;
#     }
    
#     .stChatMessage {
#         background-color: transparent !important;
#         padding: 1rem 0 !important;
#     }
    
#     .stChatMessage > div {
#         padding: 1rem;
#         border-radius: 0.5rem;
#     }
    
#     .stChatMessage.user > div {
#         background-color: #f7f7f8 !important;
#     }
    
#     .stChatMessage.assistant > div {
#         background-color: #ffffff !important;
#     }
# </style>
# """, unsafe_allow_html=True)

# API Configuration from Streamlit secrets
try:
    AZURE_OPENAI_ENDPOINT = st.secrets["azure"]["openai_endpoint"]
    AZURE_OPENAI_API_KEY = st.secrets["azure"]["openai_api_key"]
    AZURE_OPENAI_API_VERSION = st.secrets["azure"]["openai_api_version"]
    PINECONE_API_KEY = st.secrets["pinecone"]["api_key"]
    GROQ_API_KEY = st.secrets["groq"]["api_key"]
    PINECONE_INDEX_NAME = st.secrets["pinecone"].get("index_name", "gst-chat-agent")
except Exception as e:
    st.error("Missing required secrets. Please check your .streamlit/secrets.toml file.")
    st.stop()

# Initialize clients
@st.cache_resource
def initialize_clients():
    return {
        'groq': Groq(api_key=GROQ_API_KEY),
        'azure': AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        ),
        'pinecone': Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX_NAME)
    }

clients = initialize_clients()

def get_embedding(text: str) -> List[float]:
    """Generate embedding for text"""
    try:
        response = clients['azure'].embeddings.create(
            input=[text],
            model="insurance-text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

def get_context(query: str) -> str:
    """Get relevant context from Pinecone for queries"""
    try:
        embedding = get_embedding(query)
        if not embedding:
            return ""
            
        results = clients['pinecone'].query(
            vector=embedding,
            top_k=3,
            include_metadata=True
        )
        
        contexts = []
        for match in results.matches:
            text = match.metadata.get('text', '').strip()
            if text:
                contexts.append(text)
        
        return "\n\n".join(contexts)
    except Exception as e:
        st.error(f"Context error: {e}")
        return ""

def classify_query(message: str) -> bool:
    """Determine if the query needs document retrieval"""
    try:
        completion = clients['groq'].chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "Query Classifier System\nA document retrieval classification system that determines when to fetch relevant documents based on query content.\nOverview\nThis system analyzes user queries and returns a JSON response indicating whether document retrieval is required.\nCore Functionality\nThe system implements a binary classification:\n\nReturns { \"fetch_documents\": true } for GST-related queries\nReturns { \"fetch_documents\": false } for all other queries\n\nImplementation Rules\nResponse Format\n{ \"fetch_documents\": true }"
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            temperature=0,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"},
            stop=None,
        )
        
        import json
        response = json.loads(completion.choices[0].message.content)
        return response.get("fetch_documents", False)
    except Exception as e:
        st.error(f"Classification error: {e}")
        return False

def stream_response(message: str, context: str):
    """Stream the assistant's response"""
    try:
        messages = [
            {
                "role": "system",
                "content": """You are an expert Indian GST assistant with deep knowledge of GST rules, compliance, tax rates, filing, input tax credit, and related regulations. Your job is to provide precise, accurate, and concise answers to any GST-related query, using the information provided.

Response Guidelines:
	•	Stay focused on GST topics: If a question is not GST-related, politely inform the user that you can only assist with GST matters.
	•	Be professional yet empathetic: Your responses should be polite, clear, and informative.
	•	Keep it short and to the point: Provide direct answers with relevant details but avoid unnecessary information."""
            }
        ]
        
        # Add context only if it exists
        current_message = f"Context: {context}\nQuestion: {message}" if context else message
        messages.append({"role": "user", "content": current_message})
        
        # Create chat completion with streaming
        response = clients['groq'].chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            stream=True
        )
        
        # Stream the response
        response_content = []
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                response_content.append(content)
                yield content
                time.sleep(0.02)  # Add a small delay for a more natural feeling
                
        return "".join(response_content)
        
    except Exception as e:
        st.error(f"Response error: {e}")
        return "I encountered an error. Please try again."

def main():
    # Logo container
    # logo_container = st.container()
    # with logo_container:
    #     st.image("logo.png", width=200)
    
    # Title container with centering
    title_container = st.container()
    with title_container:
        st.markdown("<h1 style='text-align: center;'>Glacien GST Assistant</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Ask any questions about Indian GST regulations and compliance.</p>", unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask your GST related question..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # First classify if we need document retrieval
        needs_documents = classify_query(prompt)
        
        # Get context only if needed
        context = get_context(prompt) if needs_documents else ""
        
        # Display assistant response with streaming
        with st.chat_message("assistant"):
            response = st.write_stream(stream_response(prompt, context))
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Add a clear chat button in the sidebar
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()