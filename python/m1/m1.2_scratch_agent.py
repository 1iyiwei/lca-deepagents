from deepagents import create_deep_agent

from models import model
from my_utils import print_content
from langchain.chat_models import init_chat_model

agent = create_deep_agent(model=model)

result = agent.invoke({"messages": [{"role": "user", "content": "What is an LLM?"}]})

print_content(model, result["messages"][-1])
