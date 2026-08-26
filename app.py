"""
Optional simple chat UI for the FAQ Chatbot, built with Streamlit.

Run with:
    streamlit run app.py
"""

import streamlit as st
from chatbot import FAQChatbot

st.set_page_config(page_title="FAQ Chatbot", page_icon="💬")
st.title("💬 FAQ Chatbot")
st.caption("Ask a question and I'll match it to the closest FAQ.")

# Cache the bot so FAQs aren't re-vectorized on every rerun
@st.cache_resource
def load_bot():
    return FAQChatbot("faqs.json")

bot = load_bot()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
user_input = st.chat_input("Type your question here...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    answer, matched_q, score = bot.get_response(user_input)

    with st.chat_message("assistant"):
        st.write(answer)
        if matched_q:
            st.caption(f"Matched FAQ: \"{matched_q}\" · similarity {score:.2f}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
