from dotenv import load_dotenv
import os

import streamlit as st
from urllib.request import urlopen
from PIL import Image, ImageDraw, ImageOps
from typing import Set
from backend.core import run_llm

load_dotenv(override=True)

def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = sorted(source_urls)
    # sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

def round_corners(image, radius):
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), image.size], radius=radius, fill=255)
    result = ImageOps.fit(image, mask.size, centering=(0.5, 0.5))
    result.putalpha(mask)

    return result

# Define which model to use
MODEL = os.getenv("MODEL")

# Page Configuration
st.set_page_config(
    page_title="MyBot", 
    page_icon="🤖", 
    layout="wide"
    )


# Initialise sessions states
if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []
    

# Configure the sidebar
with st.sidebar:
    # Configure sidebar style
    st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stElementContainer] [data-testid=stImageContainer]{
            text-align: center;
            justify-content: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
        button[kind="primary"] {
            background-color: orange;
            border-color: orange;
            text-align: center;
            margin: auto;
            display: flex;
            justify-content: center;
            
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar Logo section      
    st.sidebar.image("img/bot_image.png", width=150)
    st.markdown("<h1 style='text-align: center;'>#RagBot</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar About section
    st.markdown("## What is it?")
    st.markdown(
    "This is a chatbot example that uses RAG to search and sumarise the data ingested into a Vector Database. "
    "It also implement memory of previous questions and add them to the context."
            )
    st.markdown(
    "This tool is a work in progress. "
            )

# Creates a chatbot greeting message
if "messages" not in st.session_state:
    user_name = "Rodrigo"
    st.session_state["messages"] = [
        {"role": "assistant", "content": f"Hi {user_name}, I'm the RAG chatbot. How can I help you?"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    

# Chatbot input
if prompt := st.chat_input(placeholder="Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.spinner("Generating response..."):
        
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )
        # # Run the model
        # if MODEL == "bedrock":
        #     generated_response = run_llm_bedrock(
        #         query=prompt, chat_history=st.session_state["chat_history"]
        #     )
        # elif MODEL == "openai":
        #     generated_response = run_llm_openai(
        #         query=prompt, chat_history=st.session_state["chat_history"]
        #     )
        # else:
        #     raise ValueError("Invalid model selected")
        
        # Extract sources
        sources = {doc.metadata["source"] for doc in generated_response["context"]}
        
        # Format the response
        formatted_response = (
            f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
        )
        
        # Update the session state
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["answer"]))

if st.session_state["chat_answers_history"]:
    with st.chat_message("assistant"):
        st.session_state.messages.append({"role": "assistant", "content": formatted_response})
        st.write(formatted_response)
