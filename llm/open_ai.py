import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.7

client = OpenAI()


def complete_text(prompt):
    res = client.chat.completions.create(
        messages=prompt,
        model=MODEL,
        temperature=TEMPERATURE,
    )
    return res.choices[0].message.content
