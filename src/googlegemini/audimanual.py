import os
import gradio as gr
import google.generativeai as genai
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')
LOG_FILE = 'chat.log'
log = list()
with open(LOG_FILE, 'w') as f:
    f.write('')


def ask_audi_manual(user_prompt: str, history):
    pre_instructions = ("Here is the AUDI A4 Manual I'd like you to summarize in as few words as possible:")
    with open('audia4b.md', 'r') as f:
        content = f.read()
    post_instructions = ''

    prompt = '{}\n{}\n{}\n{}'.format(pre_instructions, content, user_prompt, post_instructions)
    response = model.generate_content(prompt, stream=False)
    for chunk in response:
        yield chunk.text

    with open('chat.log', 'a') as f:
        f.write('-' * 100)
        f.write('\n')
        f.write(user_prompt)
        f.write('\n\n')
        f.writelines([chunk.text for chunk in response])
        f.write('\n\n')


gr.ChatInterface(
    ask_audi_manual,
    title="AUDI A4 Manual",
    chatbot=gr.Chatbot(height=600),
    textbox=gr.Textbox(placeholder="Ask the Audi A4 Manual?", container=False, scale=7),
).launch()


