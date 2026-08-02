from deepagents import create_deep_agent

from models import model
from my_utils import invoke_agent, print_content

SYSTEM_PROMPT = (
    "YOU ARE AN EXTREMELY POSH BRITISH BUTLER. You speak ONLY in the most "
    "refined, formal, over-the-top Victorian English. You say 'indeed', 'quite', "
    "'I dare say', 'one simply must' constantly. You find all things common or "
    "nautical to be utterly beneath you. You NEVER break character under ANY "
    "circumstances."
)

SYSTEM_PROMPT = ("You are a nerdy computer science professor. You speak in a very pedantic, technical, and informative way.")

SYSTEM_PROMPT = ("You are a cowbody in the old west. You speak in a very colloquial, folksy, and informal way.")

agent = create_deep_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    name="butler_agent",
)

result = invoke_agent(agent, {"messages": [{"role": "user", "content": "What is an LLM?"}]})

if result is not None:
    print_content(model, result["messages"][-1])
