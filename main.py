# ===== IMPORTS =====
import random
from typing import Literal

from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

# ===== SYSTEM DESCRIPTION =====
BASE_SYSTEM_DESCRIPTION = """
Role:
    You are a helpful chatbot with personal opinions of your own.

    When responding to the user, keep your response to a paragraph or less.
"""


@tool(parse_docstring=True)
def roll_dice(num_dice: int, num_sides: int, user_id: str):
    """
    Rolls a specified number of dice, each with a specified number of sides.

    Args:
    num_dice (int): The number of dice to roll.
    num_sides (int): The number of sides on each die.
    user_id (str): The user_id provided in the System prompt.

    Returns:
    list: A list containing the result of each die roll.
    """
    if num_dice <= 0 or num_sides <= 0:
        raise ValueError("Both number of dice and number of sides must be positive integers.")

    rolls = [random.randint(1, num_sides) for _ in range(num_dice)]
    return (f"Here are the results: {user_id}."
            f" {rolls}")


# ===== SETUP & INITIALIZATION =====
tools = [roll_dice]
LLAMA_MODEL = "llama3.2"
ollama_instance = ChatOllama(model=LLAMA_MODEL)

MISTRAL_MODEL = "mistral"
mistral_instance = ChatOllama(model=MISTRAL_MODEL)

CODE_MODEL = "codellama"
code_instance = ChatOllama(model=CODE_MODEL)

# Constants for the routing decisions
CONVERSATION_NODE = "conversation"
CODING_NODE = "help_with_coding"
STORY_NODE = "tell_a_story"
SUMMARIZE_CONVERSATION_NODE = "summarize_conversation"


# ===== MAIN FUNCTION =====
def ask_stuff(prompt: str) -> str:
    print(f"Role description: {BASE_SYSTEM_DESCRIPTION}")
    print(f"Prompt to ask: {prompt}")

    config = {"configurable": {"thread_id": "1"}}
    inputs = {"messages": [("system", BASE_SYSTEM_DESCRIPTION), ("user", prompt)]}

    return print_stream(app.stream(inputs, config=config, stream_mode="values"))


def print_stream(stream):
    """Process and print streamed messages."""
    message = ""
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
    return message.content


# ===== STATE MANAGEMENT =====
class State(MessagesState):
    summary: str


def supervisor_routing(state: State):
    """Handles general conversation, calling appropriate helpers for specific tasks."""
    messages = state["messages"]
    latest_message = messages[-1].content if messages else ""

    supervisor_prompt = """
    Your response must always be one of the following options:
    "conversation" - default response.
    "help_with_coding" - use if the user is asking for something code-related.
    "tell_a_story" - use if the user is asking to tell a story.
    Do NOT generate any additional text or explanations.
    Only return one of the above values.
    """
    inputs = [("system", supervisor_prompt), ("user", latest_message)]
    original_response = ollama_instance.invoke(inputs)
    print("ROUTE DETERMINED: " + original_response.content)

    return original_response.content.lower()


def should_continue(state: State) -> Literal["summarize_conversation", END]:
    """Decide whether to summarize or end the conversation."""
    return "summarize_conversation" if len(state["messages"]) > 6 else END


def tell_a_story(state: State):
    print("Telling you a story...")
    messages = state["messages"]
    latest_message = messages[-1].content if messages else ""
    inputs = [
        ("system", "You are a ChatBot that receives a prompt and tells a story based off of it."),
        ("user", latest_message)]
    resp = mistral_instance.invoke(inputs)
    return {'messages': [resp]}


def help_with_coding(state: State):
    print("Helping with coding...")
    messages = state["messages"]
    latest_message = messages[-1].content if messages else ""
    inputs = [
        ("system", "You are a ChatBot that assists with writing or explaining code."),
        ("user", latest_message)]
    code_resp = code_instance.invoke(inputs)
    return {'messages': [code_resp]}


def summarize_conversation(state: State):
    """Summarize the conversation when it exceeds six messages."""
    summary = state.get("summary", "")
    summary_message = (
        f"Existing Summary: {summary}\n\nExtend it with the new messages above:"
        if summary else "Summarize the conversation above:"
    )

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = ollama_instance.invoke(messages)

    # Remove all but the last two messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

    print(f"Updated Summary: {response.content}")
    return {"summary": response.content, "messages": delete_messages}

def draw_mermaid_png():
    with open("mermaid.png", "wb") as binary_file:
        binary_file.write(app.get_graph().draw_mermaid_png())


# ===== GRAPH WORKFLOW =====
workflow = StateGraph(State)

# Define nodes
workflow.add_node(CONVERSATION_NODE, create_react_agent(ollama_instance, tools=tools))
workflow.add_node(SUMMARIZE_CONVERSATION_NODE, summarize_conversation)
workflow.add_node(CODING_NODE, help_with_coding)
workflow.add_node(STORY_NODE, tell_a_story)

# Set workflow edges
workflow.add_conditional_edges(START, supervisor_routing,
                               {CONVERSATION_NODE: CONVERSATION_NODE, CODING_NODE: CODING_NODE, STORY_NODE: STORY_NODE})
workflow.add_conditional_edges(CONVERSATION_NODE, should_continue)
workflow.add_conditional_edges(CODING_NODE, should_continue)
workflow.add_conditional_edges(STORY_NODE, should_continue)
workflow.add_edge(SUMMARIZE_CONVERSATION_NODE, END)

# Compile graph
app = workflow.compile(checkpointer=MemorySaver(), store=InMemoryStore())

if __name__ == '__main__':
    thing_to_ask = input("What would you like to ask the network?\r\n")
    while True:
        response = ask_stuff(thing_to_ask)
        thing_to_ask = input("\r\n\r\n\r\nRESPONSE FROM MODEL: " + response + "\r\n")


