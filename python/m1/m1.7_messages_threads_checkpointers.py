from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

from models import model
from my_utils import print_content

# agent without checkpointer
agent = create_deep_agent(
    model=model,
)

# no explicit thread
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Remember that my favorite colour is blue."}]},
)
print("No explicit thread, turn 1:")
print_content(model, result["messages"][-1])

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is my favorite colour?"}]},
)
print("No explicit thread, turn 2:")
print_content(model, result["messages"][-1])

# agent with checkpointer

agent = create_deep_agent(
    model=model,
    checkpointer=MemorySaver(),
)

# threads
thread_a = {"configurable": {"thread_id": "m1-7-thread-a"}}
thread_b = {"configurable": {"thread_id": "m1-7-thread-b"}}

# thread a with 2 turns
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Remember that my favorite colour is blue."}]},
    config=thread_a,
)
print("Thread A, turn 1:")
print_content(model, result["messages"][-1])

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is my favorite colour?"}]},
    config=thread_a,
)
print("\nThread A, turn 2:")
print_content(model, result["messages"][-1])

# thread b with 1 turn
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is my favorite colour?"}]},
    config=thread_b,
)
print("\nThread B, turn 1:")
print_content(model, result["messages"][-1])
