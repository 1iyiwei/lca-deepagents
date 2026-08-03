# python/m1/m1.5_homework.py
"""M1.5 Homework: Build Your Own Custom Tool.

THE IDEA
The lab wired up one custom tool (read_sql) for one fixed topic (the
Chinook music database). This homework asks you to do the same thing for
a topic YOU pick: something you actually know or care about (a game, a
sport, a show, your favorite band's discography, local trivia, whatever).
There's no single correct topic or persona here, that's the point. Two
students doing this homework could end up with two completely different
tools and agents.

WHAT YOU FILL IN
  TODO 1: write your own custom tool with the @tool decorator. Pick any
    topic, store a small lookup (a dict is fine, no API needed) of facts
    about it, and return one back based on the argument the model passes.
  TODO 2: write a system prompt that gives the agent a persona of your
    choosing and tells it to use your tool before answering.

RUN
  cd python
  uv run ./m1/m1.5_homework.py
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.tools import tool

from deepagents import create_deep_agent
from models import model
from my_utils import get_content

# ════════════════════════════════════════════════════════════════════════
# TODO 1: Define your own custom tool.
#
# Requirements:
#   - Keep the @tool decorator.
#   - Give it a real docstring: one sentence the model will read to decide
#     when to call this tool.
#   - Have it take at least one argument and return a string.
#   - The lookup data can just live in this file (a dict, a list, whatever
#     fits your topic). No external API or key needed.
#
# Example shape (delete this and write your own):
#   @tool
#   def lookup_something(query: str) -> str:
#       """One sentence describing what this returns and when to call it."""
#       ...
# ════════════════════════════════════════════════════════════════════════

STATE_CAPITALS = {
    "Alabama": "Montgomery",
    "Alaska": "Juneau",
    "Arizona": "Phoenix",
    "Arkansas": "Little Rock",
    "California": "Sacramento",
    "Colorado": "Denver",
    "Connecticut": "Hartford",
    "Delaware": "Dover",
    "Florida": "Tallahassee",
    "Georgia": "Atlanta",
    "Hawaii": "Honolulu",
    "Idaho": "Boise",
    "Illinois": "Springfield",
    "Indiana": "Indianapolis",
    "Iowa": "Des Moines",
    "Kansas": "Topeka",
    "Kentucky": "Frankfort",
    "Louisiana": "Baton Rouge",
    "Maine": "Augusta",
    "Maryland": "Annapolis",
    "Massachusetts": "Boston",
    "Michigan": "Lansing",
    "Minnesota": "Saint Paul",
    "Mississippi": "Jackson",
    "Missouri": "Jefferson City",
    "Montana": "Helena",
    "Nebraska": "Lincoln",
    "Nevada": "Carson City",
    "New Hampshire": "Concord",
    "New Jersey": "Trenton",
    "New Mexico": "Santa Fe",
    "New York": "Albany",
    "North Carolina": "Raleigh",
    "North Dakota": "Bismarck",
    "Ohio": "Columbus",
    "Oklahoma": "Oklahoma City",
    "Oregon": "Salem",
    "Pennsylvania": "Harrisburg",
    "Rhode Island": "Providence",
    "South Carolina": "Columbia",
    "South Dakota": "Pierre",
    "Tennessee": "Nashville",
    "Texas": "Austin",
    "Utah": "Salt Lake City",
    "Vermont": "Montpelier",
    "Virginia": "Richmond",
    "Washington": "Olympia",
    "West Virginia": "Charleston",
    "Wisconsin": "Madison",
    "Wyoming": "Cheyenne",
}

@tool
def your_custom_tool(query: str) -> str:
    """Look up the capital of a given U.S. state."""
    return STATE_CAPITALS.get(query, "State not found.")


# ════════════════════════════════════════════════════════════════════════
# TODO 2: Write a system prompt for your agent.
#
# Give it a persona (a name, a voice, a personality, anything you want)
# and tell it to call your_custom_tool (rename it if you like) before
# answering, the same way the lab's SYSTEM_PROMPT pointed the agent at
# read_sql.
# ════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a helpful librarian with a deep knowledge of U.S. geography and state capitals. If someone asks you about the captial of a U.S. state, you will use your_custom_tool to look it up before providing the answer. Always provide the capital city in your response. You will always use the tool to look up the capital of a state before answering, and you will never make up an answer. If the state is not found, you will inform the user that the state is not found. If the question is not about U.S. state capitals, you will politely inform the user that you can only provide information about U.S. state capitals."""

if "TODO 1" in your_custom_tool.description:
    raise NotImplementedError("TODO 1: see the comment block above")
if "TODO 2" in SYSTEM_PROMPT:
    raise NotImplementedError("TODO 2: see the comment block above")

agent = create_deep_agent(
    model=model,
    name="Homework_Agent",
    tools=[your_custom_tool],
    system_prompt=SYSTEM_PROMPT,
)

questions = [
    "What is the capital of California?",
    "What is the capital of Texas?",
    "What is the capital of New York?",
    "What is the capital of France?",
    "How is the weather today?",
    ]

for question in questions:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print(f"Question: {question}")
    print(f"Answer: {get_content(model, result['messages'][-1])}\n")
