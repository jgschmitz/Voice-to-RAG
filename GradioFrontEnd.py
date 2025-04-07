# super basic
import gradio as gr
from rag_core import generate_rag_response
from voice_utils import transcribe_audio

def handle_input(audio):
    file_path = audio
    query = transcribe_audio(file_path)
    return generate_rag_response(query)

gr.Interface(fn=handle_input, inputs="microphone", outputs="text").launch()
