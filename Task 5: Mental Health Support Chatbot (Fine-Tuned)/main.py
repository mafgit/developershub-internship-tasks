import streamlit as st
from model import run_model

st.set_page_config(
    page_title="Empathetic Assistant",
    page_icon="❄️")

# intro
st.title("❄️ Empathetic Assistant")

with st.expander("About", icon=':material/info:', expanded=True):
    st.markdown("> This app features a fine-tuned version of `DistilGPT2` on `Estwld/empathetic_dialogues_llm` dataset.\n")
    
    # settings
    version_options = ["mafgit/empathetic-distilgpt2", "distilbert/distilgpt2"]
    version = st.selectbox("Select model version", version_options, index=0, key='version')

    st.warning("Since it is a very small model and the dataset is not enough, it is still extremely limited, although differences are clearly visible.")



# message history
if not 'messages' in st.session_state:
    st.session_state['messages'] = []

if len(st.session_state['messages'])==0:
    st.markdown('\n')
else:
    for message in st.session_state['messages']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])


# input
content = st.chat_input("Hello! How can I assist you today?")

if content:
    content = content.strip()

model = None
tokenizer = None


# inference
if content:
    st.session_state['messages'].append({
        'role': 'user',
        'content': content
    })

    st.chat_message("user").markdown(content)

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):

            response, was_truncated = run_model(version, st.session_state['messages'])
            st.session_state['messages'].append({
                'role': 'assistant',
                'content': response
            })

            
            if was_truncated:
                st.markdown("**...**", help="The response was truncated to keep context intact for a little longer.")



# context window usage
if not 'context_window' in st.session_state:
    st.session_state['context_window'] = 0.0

st.progress(st.session_state['context_window'], text=f"📊 Context window usage: **{st.session_state['context_window']:.2%}** (max 1024 tokens in window)")