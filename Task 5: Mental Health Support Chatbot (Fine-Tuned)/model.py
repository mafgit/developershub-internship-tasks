
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import streamlit as st
from typing import List, Dict
import threading

@st.cache_resource(show_spinner=False)
def load_model(version: str):
    with st.spinner(f"Loading {version}..."):
        model = AutoModelForCausalLM.from_pretrained(version)
        tokenizer = AutoTokenizer.from_pretrained(version)
        # pipe = pipeline('text-generation', model=version, tokenizer=version)

    return model, tokenizer


def format_conversation(messages: List[Dict]) -> str:
    text = ''
    
    for msg in messages[-5:]:
        if msg['role'] == 'user':
            text += f'[USER] {msg["content"]} '
        else:
            # if msg['role'] == 'assistant':
            text += f'[ASSISTANT] {msg["content"]}<|endoftext|>'
        
    
    text += '[ASSISTANT] '
    return text


model = None
tokenizer = None

context_limit = 1024
max_new_tokens = 384
max_input_tokens = 640

def run_model(version, messages: List[Dict]):
    global model, tokenizer

    model, tokenizer = load_model(version)

    formatted = format_conversation(messages)

    inputs = tokenizer(formatted, return_tensors='pt', truncation=True, max_length=max_input_tokens)
    
    with st.expander('Context Passed'):
        st.text(tokenizer.decode(inputs.input_ids[0]))
    
    prompt_token_count = inputs.input_ids.shape[1]

    full_text = ''

    def stream_generator():
        nonlocal full_text
        global tokenizer, model

        streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True) # pyright: ignore
            
        kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            streamer=streamer,
            repetition_penalty=1.2
        )

        thread = threading.Thread(target=model.generate, kwargs=kwargs) # pyright: ignore
        thread.start()

        if streamer is not None:
            for text in streamer:
                full_text += text
                yield text
                if text == '<|endoftext|>':
                    break

        
    
    st.write_stream(stream_generator, cursor="❄️")



    output_token_count = len(tokenizer.encode(full_text))
    st.session_state['context_window'] = min((output_token_count + prompt_token_count) / context_limit, 1.0)

    was_truncated = False
    if output_token_count == max_new_tokens:
        was_truncated = True

    return full_text, was_truncated
