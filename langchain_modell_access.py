from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Claude-3 Sonnet を使用するように設定
llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0,
    max_tokens=1000
)

messages = [
    SystemMessage(content="You are a helpful assistant that can answer questions."),
    HumanMessage(content="What is the capital of the moon?"),
]

response = llm.invoke(messages)

print(response.content)
