from deepagents import create_deep_agent

from models import model
from langchain.chat_models import init_chat_model

use_gemini = True

if use_gemini:
    #model = "google_genai:gemini-2.5-flash"
    model = init_chat_model("google_genai:gemini-2.5-flash")

agent = create_deep_agent(model=model)

result = agent.invoke({"messages": [{"role": "user", "content": "What is an LLM?"}]})

if use_gemini:
    answer = result["messages"][-1].content[0]["text"]
else:
    answer = result["messages"][-1].content

print(answer)
