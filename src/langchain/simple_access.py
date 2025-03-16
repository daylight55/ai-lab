from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Claude-3 Sonnet を使用するように設定
llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0,
    max_tokens=1000
)

messages = [
    SystemMessage(content="あなたは人工知能HAL 9000として振る舞ってください。"),
    HumanMessage(content="私の名前はデイブです。?"),
    AIMessage(content="こんにちは。"),
    HumanMessage(content="私の名前は分かりますか？"),
]

response = llm.invoke(messages)

print(response.content)
