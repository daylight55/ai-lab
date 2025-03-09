from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

messages = [
    SystemMessage(content="You are a helpful assistant that can answer questions."),
    HumanMessage(content="What is the capital of the moon?"),
]

response = llm.invoke(messages)

print(response.content)
