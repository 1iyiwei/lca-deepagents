from deepagents import create_deep_agent

from models import model
from my_utils import invoke_agent, print_content

agent = create_deep_agent(model=model)

result = invoke_agent(agent, {"messages": [{"role": "user", "content": "What is an LLM?"}]})

if result is not None:
    print_content(model, result["messages"][-1])
