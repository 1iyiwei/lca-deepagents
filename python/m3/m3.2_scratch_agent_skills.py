from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend

from models import model
from my_utils import print_content

m3_dir = Path(__file__).parent
backend = FilesystemBackend(root_dir=str(m3_dir), virtual_mode=True)

agent = create_deep_agent(
    model=model,
    name="Sales_Assistant",
    backend=backend,
    skills=["/skills"],
    system_prompt="You are a sales assistant.",
)

QUALIFY_LEAD_MESSAGE = "Qualify this lead: Acme Corp, 200-person logistics company. I spoke with Sarah Chen, VP of Sales: she's the decision maker. They have $45k budgeted for CRM this year. Main pain: deals are slipping through the cracks due to poor pipeline visibility. They want a solution live by end of Q3."

DRAFT_PITCH_MESSAGE="Write a cold outreach message for a prospect at a mid-size logistics company."

for message in [QUALIFY_LEAD_MESSAGE, DRAFT_PITCH_MESSAGE]:
    # print a line for separation
    print("-"*16)
    result = agent.invoke({"messages": [{"role": "user", "content": message}]})
    print_content(model, result["messages"][-1])
