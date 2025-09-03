import streamlit as st
from huggingface_hub import InferenceClient
import os
import re
from dotenv import load_dotenv

# Load token from .env file
load_dotenv()

# Initialize Hugging Face client
client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"], 
)

# Set Streamlit app config
st.set_page_config(page_title="Smol Chatbot", layout="centered")
st.title("🧠 Smol Chatbot")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Input box for new question
user_input = st.chat_input("Ask something...")

if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    # Query the model
    with st.spinner("Thinking..."):
        try:
            response = client.chat.completions.create(
                model="HuggingFaceTB/SmolLM3-3B",
                messages=st.session_state.chat_history,
            )

            raw_answer = response.choices[0].message.content.strip()

            #  Remove <think>...</think> content using regex
            cleaned_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()

        except Exception as e:
            cleaned_answer = f"⚠️ Error: {e}"

    # Add assistant's response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": cleaned_answer})
    st.chat_message("assistant").markdown(cleaned_answer)

