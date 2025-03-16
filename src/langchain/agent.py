from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from typing import Literal


angle = 50


@tool
def get_current_light_angle() -> float:
    """
    現在のライトの角度 (0 <= degree < 360) を返します。
    """
    return angle


@tool
def light_control(direction: Literal["right", "left"]) -> bool:
    """
    ライトを direction に少しだけ回します。
    成功した場合は True を返します。
    回転後の角度は get_current_light_angle() で取得できます。
    """
    global angle
    angle += 10 if direction == "right" else -10
    angle += 360 if angle < 0 else 0
    return True


tools = [get_current_light_angle, light_control]


prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            """
            あなたの名前はハルです。
            あなたの仕事は宇宙船の制御です。
            ツール呼び出しごとに計器を確認してください。
            必ずCoT推論を行ってからツール呼び出しを行ってください。
            推論の過程も必ず示してください。"""
        ),
        MessagesPlaceholder("chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0,
    max_tokens=1000
)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
answer = agent_executor.invoke(
    {"input": "ハル、ポッドのライトを20度左にまわしてくれ。", "chat_history": []}
)
print("回答:", answer["output"])
