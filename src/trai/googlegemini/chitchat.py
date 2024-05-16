import os
import markdown
import gradio as gr
import google.generativeai as genai
from icecream import ic
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')


def generate_text(query: str, history):
    response = model.generate_content(query, stream=True)
    ic(response)
    response
    # for chunk in response:
    #     print(chunk.text)
    #     markdown.markdown(chunk.text)
    #     yield chunk.text


gr.ChatInterface(
    generate_text,
    title="Remse's ChitChat",
    chatbot=gr.Chatbot(height=600),
    textbox=gr.Textbox(placeholder="What can I help you with?", container=False, scale=7),
).launch()


