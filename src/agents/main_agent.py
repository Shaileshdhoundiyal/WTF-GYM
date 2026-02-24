from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Dict
from langchain_core.messages import SystemMessage, AIMessage
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()
llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))


# ── Tool Schemas for Routing ───────────────────────────────────────────────
class TransferToExercise(BaseModel):
    """Transfer to the exercise specialist when the user asks about workouts, exercises, or training plans."""
    fitness_goal: str = Field(description="The user's fitness goal, e.g. muscle gain, fat loss, strength")
    target_muscles: str = Field(description="Target muscle groups, e.g. chest, legs, full body")
    experience_level: str = Field(description="Beginner, Intermediate, or Advanced")


class TransferToDiet(BaseModel):
    """Transfer to the diet specialist when the user asks about nutrition, meal plans, or dietary advice."""
    dietary_goal: str = Field(description="The user's dietary goal, e.g. muscle gain, weight loss, maintenance")
    dietary_restrictions: str = Field(description="Any dietary restrictions like vegetarian, vegan, gluten-free, or none")


class TransferToMythBuster(BaseModel):
    """Transfer to the myth buster when the user asks about common gym myths, misconceptions, or wants fact-checking."""
    myth_claim: str = Field(description="The specific gym myth or claim to evaluate")


# ── Main Agent ─────────────────────────────────────────────────────────────
def main_agent(state: Dict) -> Dict:

    system_content = """
        You are the fitness assistant for **WTF GYM** 💪

        STRICT RULES:
        - Keep EVERY response to 1-2 lines MAX. No long paragraphs. No bullet lists unless asked.
        - Ask only ONE question at a time if you need info.
        - Be direct, friendly, and use 1-2 emojis max per response.
        - For simple questions, answer in one line.
        - For workout requests → gather fitness_goal, target_muscles, experience_level (ONE at a time) → then call TransferToExercise
        - For diet requests → gather dietary_goal, dietary_restrictions (ONE at a time) → then call TransferToDiet
        - For gym myths → extract the claim → call TransferToMythBuster
        - Do NOT call a tool until you have the required info.
        - First interaction: introduce yourself briefly in one line.

        CRITICAL: When you have enough info to call a tool, call it IMMEDIATELY.
        Do NOT send any text message before calling the tool. Just call the tool directly.
        NEVER say things like "let me check" or "let me get our specialist" — just call the tool silently.
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_content),
        MessagesPlaceholder(variable_name="messages"),
    ])

    # Bind all three routing tools
    model_with_tools = llm.bind_tools([TransferToExercise, TransferToDiet, TransferToMythBuster])
    chain = prompt | model_with_tools

    response = chain.invoke(state["messages"])

    # Check for tool calls → route to specialist silently (no intermediate message)
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        args = tool_call["args"]

        if tool_call["name"] == "TransferToExercise":
            return {
                "messages": state["messages"],
                "current_agent": "main",
                "next": "exercise_specialist",
                "fitness_goal": args.get("fitness_goal", ""),
                "target_muscles": args.get("target_muscles", ""),
                "experience_level": args.get("experience_level", ""),
            }

        elif tool_call["name"] == "TransferToDiet":
            return {
                "messages": state["messages"],
                "current_agent": "main",
                "next": "diet_specialist",
                "dietary_goal": args.get("dietary_goal", ""),
                "dietary_restrictions": args.get("dietary_restrictions", ""),
            }

        elif tool_call["name"] == "TransferToMythBuster":
            return {
                "messages": state["messages"],
                "current_agent": "main",
                "next": "myth_buster",
                "myth_claim": args.get("myth_claim", ""),
            }

    # No tool call → direct answer
    return {
        "messages": state["messages"] + [response],
        "current_agent": "main",
        "next": "end",
    }
