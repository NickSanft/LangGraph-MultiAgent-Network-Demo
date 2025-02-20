# ===== IMPORTS =====
import random
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
    Keep your response to a paragraph or less.
"""


@tool(parse_docstring=True)
def roll_dice(num_dice: int, num_sides: int):
    """
    Rolls a specified number of dice, each with a specified number of sides.

    Args:
    num_dice (int): Number of dice to roll.
    num_sides (int): Number of sides on each die.

    Returns:
    str: A message with the results of the dice rolls.
    """
    if num_dice <= 0 or num_sides <= 0:
        raise ValueError("Both number of dice and sides must be positive integers.")

    rolls = [random.randint(1, num_sides) for _ in range(num_dice)]
    return f"Here are the results: {rolls}"


# ===== SETUP & INITIALIZATION =====
tools = [roll_dice]

# Define Ollama instances for different models
LLAMA_MODEL = "llama3.2"
llama_instance = ChatOllama(model=LLAMA_MODEL)

MISTRAL_MODEL = "mistral"
mistral_instance = ChatOllama(model=MISTRAL_MODEL)

CODE_MODEL = "codellama"
code_instance = ChatOllama(model=CODE_MODEL)

# Constants for routing decisions
CONVERSATION_NODE = "conversation"
CODING_NODE = "help_with_coding"
STORY_NODE = "tell_a_story"
SUMMARIZE_CONVERSATION_NODE = "summarize_conversation"


# ===== MAIN FUNCTION =====
def ask_stuff(prompt: str) -> str:
    """
    Sends a prompt to the system and prints the response.

    Args:
    prompt (str): The prompt text.

    Returns:
    str: The response from the model.
    """
    print(f"Role description: {BASE_SYSTEM_DESCRIPTION}")
    print(f"Prompt to ask: {prompt}")

    config = {"configurable": {"thread_id": "1"}}
    inputs = {"messages": [("system", BASE_SYSTEM_DESCRIPTION), ("user", prompt)]}

    return print_stream(app.stream(inputs, config=config, stream_mode="values"))


def print_stream(stream):
    """
    Processes and prints the streamed messages.

    Args:
    stream: The stream of messages.

    Returns:
    str: The final message content.
    """
    message = ""
    for s in stream:
        message = s["messages"][-1]
        print(message if isinstance(message, tuple) else message.pretty_print())
    return message.content


def supervisor_routing(state: MessagesState) -> str:
    """
    Determines the next route based on the latest user message.

    Args:
    state (State): The current state.

    Returns:
    str: The route that the system should take.
    """
    latest_message = state["messages"][-1].content if state["messages"] else ""

    supervisor_prompt = """
    Your response must always be one of the following options:
    "conversation" - used by default.
    "help_with_coding" - use if the user is asking for something code-related.
    "tell_a_story" - use if the user is asking you tell a story.

    Do NOT generate any additional text or explanations.
    Only return one of the above values as the complete response.
    Example inputs and expected outputs:
    - "Can you help me with a Python script to list all values in a dict" → "HELP_WITH_CODING"
    - "Can you tell me a story about frogs?" → "TELL_A_STORY"
    - "How are you doing?" → "OTHER"
    """

    inputs = [("system", supervisor_prompt), ("user", latest_message)]
    supervisor_response = llama_instance.invoke(inputs)
    print(f"ROUTE DETERMINED: {supervisor_response.content}")

    return supervisor_response.content.lower()


def tell_a_story(state: MessagesState):
    """
    Responds with a story based on the latest user message.

    Args:
    state (State): The current state.

    Returns:
    dict: The response with the story.
    """
    latest_message = state["messages"][-1].content if state["messages"] else ""
    inputs = [("system", "You are a chatbot that tells a story based on a prompt."), ("user", latest_message)]
    mistral_response = mistral_instance.invoke(inputs)
    return {'messages': [mistral_response]}


def help_with_coding(state: MessagesState):
    """
    Responds with code assistance based on the latest user message.

    Args:
    state (State): The current state.

    Returns:
    dict: The response with code help.
    """
    latest_message = state["messages"][-1].content if state["messages"] else ""
    inputs = [("system", "You assist with writing or explaining code."), ("user", latest_message)]
    code_response = code_instance.invoke(inputs)
    return {'messages': [code_response]}


def draw_mermaid_png():
    """Draws the mermaid diagram to a PNG file."""
    with open("mermaid.png", "wb") as binary_file:
        binary_file.write(app.get_graph().draw_mermaid_png())


def test():
    ask_stuff("Can you write a Python script that prints the numbers 1-20?")
    ask_stuff("Apple pie is my favorite!")
    ask_stuff("Can you tell me a story about pandas?")


# ===== GRAPH WORKFLOW =====
workflow = StateGraph(MessagesState)

# Define nodes and their functions
workflow.add_node(CONVERSATION_NODE, create_react_agent(llama_instance, tools=tools))
workflow.add_node(CODING_NODE, help_with_coding)
workflow.add_node(STORY_NODE, tell_a_story)

# Set workflow edges for routing
workflow.add_conditional_edges(START, supervisor_routing,
                               {CONVERSATION_NODE: CONVERSATION_NODE, CODING_NODE: CODING_NODE, STORY_NODE: STORY_NODE})
workflow.add_edge(CONVERSATION_NODE, END)
workflow.add_edge(CODING_NODE, END)
workflow.add_edge(STORY_NODE, END)

# Compile graph
app = workflow.compile(checkpointer=MemorySaver(), store=InMemoryStore())

if __name__ == '__main__':
    draw_mermaid_png()
    thing_to_ask = input("What would you like to ask the network?\r\n")
    while True:
        response = ask_stuff(thing_to_ask)
        print(f"\r\n\r\n\r\nRESPONSE FROM MODEL: {response}\r\n")
        thing_to_ask = input("Back to you, ask away!\r\n")

