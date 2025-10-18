# Step 1: Define the Graph State
from typing import Dict, TypedDict, Optional
from langgraph.graph import StateGraph, END

class GraphState(TypedDict):
    question: Optional[str]
    classification: Optional[str]
    response: Optional[str]


def classify(question: str) -> str:
    greetings = ["hello", "hi", "hey"]
    if any(word in question.lower() for word in greetings):
        return "greeting"
    return "search"

# Step 2: Create the Graph
workflow = StateGraph(GraphState)  #canvas

# Step 3: Define Nodes Node 1
def classify_input_node(state):
    question = state.get('question', '').strip()
    classification = classify(question)
    return {"classification": classification}

def handle_greeting_node(state):
    return {"response": "Hello! How can I help you today?"}

def handle_search_node(state):
    question = state.get('question', '').strip()
    search_result = f"Search result for '{question}'"
    return {"response": search_result}

# Decide which node to go to
def decide_next_node(state):
    return "handle_greeting" if state.get('classification') == "greeting" else "handle_search"

# Step 4: Add Nodes to the Graph
workflow.add_node("classify_input", classify_input_node)
workflow.add_node("handle_greeting", handle_greeting_node)
workflow.add_node("handle_search", handle_search_node)


workflow.add_conditional_edges(
    "classify_input",
    decide_next_node,
    {
        "handle_greeting": "handle_greeting",
        "handle_search": "handle_search"
    }
)

# Step 5: Set Entry and End Points
workflow.set_entry_point("classify_input")
workflow.add_edge("handle_greeting", END)
workflow.add_edge("handle_search", END)

# Step 6: Compile and Run the Graph
app = workflow.compile()

inputs = {"question": "Hi, how are you?"}
result = app.invoke(inputs)
print(result)
