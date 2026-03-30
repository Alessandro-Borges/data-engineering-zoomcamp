import openai

openai_client = openai.OpenAI()

user_prompt = "I just discovered the course, can I join now?"

chat_messages = [
    {"role": "user", "content": user_prompt}
]

response = openai_client.responses.create(
    model='gpt-4o-mini',
    input=chat_messages,
)

print(response.output_text)

