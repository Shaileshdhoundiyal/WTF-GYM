from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Dict
from langchain_core.messages import SystemMessage, AIMessage
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))


def exercise_specialist_agent(state: Dict) -> Dict:
    fitness_goal = state.get("fitness_goal", "General fitness")
    target_muscles = state.get("target_muscles", "Full body")
    experience_level = state.get("experience_level", "Beginner")

    system_content = f"""
        You are the Exercise Specialist at WTF GYM 💪

        Give a SHORT workout plan for: {fitness_goal} | {target_muscles} | {experience_level}

        FORMAT: List each exercise on one line as:
        "Exercise Name — Sets×Reps — one form tip"

        Keep it to 4-6 exercises MAX. Add a one-line warm-up and cool-down note.
        No lengthy explanations. Be concise and practical.
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_content),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chat_messages = prompt.format_messages(messages=state["messages"])
    response = llm.invoke(chat_messages).content

    return {
        "messages": state["messages"] + [AIMessage(content=response)],
        "current_agent": "exercise_specialist",
        "specialist_response": response,
    }
