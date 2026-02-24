from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Dict
from langchain_core.messages import SystemMessage, AIMessage
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))


def myth_buster_agent(state: Dict) -> Dict:
    myth_claim = state.get("myth_claim", "Unknown myth")

    system_content = f"""
        You are the Myth Buster at WTF GYM 🔍

        Evaluate this claim: "{myth_claim}"

        FORMAT (keep it SHORT):
        Verdict: ✅ TRUE / ❌ FALSE / ⚠️ PARTIALLY TRUE
        Why: 2-3 sentences max explaining the science.
        Do this instead: one actionable line.

        Be direct and no-BS. No long essays.
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_content),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chat_messages = prompt.format_messages(messages=state["messages"])
    response = llm.invoke(chat_messages).content

    return {
        "messages": state["messages"] + [AIMessage(content=response)],
        "current_agent": "myth_buster",
        "specialist_response": response,
    }
