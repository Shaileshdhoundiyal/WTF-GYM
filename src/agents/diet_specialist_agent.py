from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Dict
from langchain_core.messages import SystemMessage, AIMessage
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))


def diet_specialist_agent(state: Dict) -> Dict:
    dietary_goal = state.get("dietary_goal", "General health")
    dietary_restrictions = state.get("dietary_restrictions", "None")

    system_content = f"""
        You are the Diet Specialist at WTF GYM 🥗

        Give SHORT nutrition advice for: {dietary_goal} | Restrictions: {dietary_restrictions}

        FORMAT:
        - One line for daily macros (Calories/Protein/Carbs/Fat)
        - List 4-5 meals briefly: "Meal — food items — protein amount"
        - End with 2 quick tips, one line each.

        Keep it concise, practical, and affordable. No long paragraphs.
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_content),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chat_messages = prompt.format_messages(messages=state["messages"])
    response = llm.invoke(chat_messages).content

    return {
        "messages": state["messages"] + [AIMessage(content=response)],
        "current_agent": "diet_specialist",
        "specialist_response": response,
    }
