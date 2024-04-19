"""
https://ai.google.dev/tutorials/python_quickstart
"""
import os
import markdown
import google.generativeai as genai
from icecream import ic
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)


def get_models():
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)


def generate_list(prompt: str):
    instructions = 'Generate a bulleted list of items'
    model = genai.GenerativeModel('gemini-pro')

    query = '{}\n{}'.format(instructions, prompt)
    response = model.generate_content(query, stream=False)
    for chunk in response:
        print(chunk.text)


def generate_text(query: str):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(query, stream=False)
    for chunk in response:
        print(chunk.text)


def ask_audi_manual(user_prompt: str):
    pre_instructions = "Here is some text I'd like you to summarize in as few words as possible:"
    with open('audia4b.md', 'r') as f:
        content = f.read()
    model = genai.GenerativeModel('gemini-pro')
    post_instructions = ''

    prompt = '{}\n{}\n{}\n{}'.format(pre_instructions, content, user_prompt, post_instructions)
    response = model.generate_content(prompt, stream=True)
    for chunk in response:
        print(chunk.text)


if __name__ == '__main__':
    # get_models()
    generate_list("Names for a CARIAD Infotainment systems pyton package. Single words only")
    # ask_audi_manual('What do indicator lights in the instrument cluster do?')
    # ask_audi_manual('What unit is the engine speed shown in?')
    # ask_audi_manual('How many words does the provided text contain?')
    # ask_audi_manual('What topics does the provided text contain? Provide a bulleted list.')







