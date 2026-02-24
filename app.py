import os
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from src.graph import app_graph

load_dotenv()


# ── FastAPI Application ────────────────────────────────────────────────────
app = FastAPI(
    title="WTF GYM Chatbot",
    description="Agentic AI chatbot for WTF GYM — exercises, diet, and myth busting",
    version="1.0.0",
)


@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for the WTF GYM chatbot."""

    # Initialize conversation state
    state = {
        "messages": [],
        "current_agent": "main",
        "next": "",
        "fitness_goal": "",
        "target_muscles": "",
        "experience_level": "",
        "dietary_goal": "",
        "dietary_restrictions": "",
        "myth_claim": "",
        "specialist_response": "",
    }

    try:
        await websocket.accept()
        print("🏋️ WTF GYM — WebSocket connection accepted")
    except Exception as e:
        print(f"Error accepting WebSocket connection: {e}")
        return

    try:
        while True:
            print("Waiting for user message...")
            data = await websocket.receive_json()

            if "message" not in data:
                await websocket.send_json({"error": "No message provided"})
                continue

            # Add user message to state
            state["messages"].append(HumanMessage(content=data["message"]))

            # Reset routing signal so each turn starts fresh
            state["next"] = ""

            # Run the graph
            result = app_graph.invoke(state)

            # Extract the AI response (last message)
            ai_message = result["messages"][-1].content

            # Send response back to user
            await websocket.send_json({"response": ai_message})

            # Update local state for next turn
            state = result

    except WebSocketDisconnect as e:
        print(f"Client disconnected: {e}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    print(f"🏋️ WTF GYM Chatbot starting on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
