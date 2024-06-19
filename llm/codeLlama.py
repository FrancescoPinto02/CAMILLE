from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


def complete_text(prompt):
    res = client.chat.completions.create(
        messages=prompt,
        model="TheBloke/CodeLlama-13B-Instruct-GGUF",
        temperature=0.5,
    )
    return res.choices[0].message.content
