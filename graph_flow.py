from langgraph.graph import StateGraph

def retrieval_node(state):
    return state

def explanation_node(state):
    return state

def decision_node(state):
    # Example: if low confidence → flag warning
    if state.get("confidence", 0) < 0.5:
        state["warning"] = "Low confidence prediction"
    else:
        state["warning"] = None
    return state

builder = StateGraph(dict)

builder.add_node("retrieve", retrieval_node)
builder.add_node("explain", explanation_node)
builder.add_node("decision", decision_node)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "explain")
builder.add_edge("explain", "decision")

graph = builder.compile()