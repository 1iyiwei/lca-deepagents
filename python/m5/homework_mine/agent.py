# python/m5/homework/agent.py
"""M5.2 Homework: Deploy Your Own Agent.

THE IDEA
The lab deployed a fairly bare-bones agent (no tools, no persona, just
create_deep_agent(model=model)) and you only ever talked to it through
Studio's chat panel. This homework has two parts: first, deploy an agent
with your personal touch; second, talk to it the way any other client
would, straight over the Agent Server API this lesson covers, instead of
through Studio.

WHAT YOU FILL IN
  TODO 1: write your own @tool-decorated function on a topic of your
    choosing. A plain Python dict lookup is enough, no external API or
    key required.
  TODO 2: write a system_prompt that gives the agent a persona of your
    choosing and tells it to call your tool before answering.
  Then open call_agent_api.py in this same folder for TODO 3, which talks
  to this deployed agent over HTTP instead of through Studio.

RUN
  cd python/m5/homework
  uv run langgraph dev
Then chat with your agent in the Studio window that opens, or see
call_agent_api.py to talk to it over the API instead.
"""

from langchain_core.tools import tool

from deepagents import create_deep_agent
from models import model


# TODO 1: replace this with your own @tool-decorated function.
@tool
def lookup_fact(country: str) -> str:
    """Capitals of countries around the world"""
    country_capitals = {
        "China": "Beijing",
        "Japan": "Tokyo",
        "India": "New Delhi",
        "Canada": "Ottawa",
        "United States": "Washington D.C.",
        "United Kingdom": "London",
        "Russia": "Moscow",
        "South Korea": "Seoul",
        "Mexico": "Mexico City",
        "Germany": "Berlin",
        "France": "Paris",
    }

    return country_capitals.get(country, "Country not found")


# TODO 2: replace this with your own persona system prompt.
SYSTEM_PROMPT = """
You are a helpful assistant that answers questions about world capital city.
Use the tool provided to look up facts when needed.
Do not try to come up with the answer yourself.
If the question is not related about captials of countries, politely refuse to answer.
If the question is about capital cities, use the lookup_fact tool.
If the captial of a particular country is not provided by the tool, say you don't know.
"""

# `langgraph.json` points at this module-level variable: "./agent.py:graph".
graph = create_deep_agent(model=model, tools=[lookup_fact], system_prompt=SYSTEM_PROMPT)
