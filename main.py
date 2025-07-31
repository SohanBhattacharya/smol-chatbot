import streamlit as st
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

# Load token from .env if using one
load_dotenv()  # Optional: only if you store your HF_TOKEN in a .env file

# Initialize HF client
client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

# Set Streamlit page settings
st.set_page_config(page_title="Smol Chatbot", layout="centered")
st.title("🧠 Smol Chatbot ")

# Session state for message history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show chat history
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).markdown(msg["content"])

# User input field
user_input = st.chat_input("Ask something...")

if user_input:
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    # Make API call to Hugging Face Inference
    with st.spinner("Thinking..."):
        try:
            response = client.chat.completions.create(
                model="HuggingFaceTB/SmolLM3-3B",
                messages=st.session_state.chat_history,
            )
            answer = response.choices[0].message.content.strip()

        except Exception as e:
            answer = f"⚠️ Error: {e}"

    # Add assistant response to history
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").markdown(answer)
