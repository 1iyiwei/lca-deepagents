# python/m3/m3.2_homework.py
"""M3.2 Homework: Bundle a Reference File Into Your Skill.

THE IDEA
The lab's two skills (qualify-lead and draft-pitch) are each a single flat
SKILL.md file with everything inline. But the lesson also covered a third
stage of progressive disclosure: a skill can point to supporting files (a
reference doc, a template, a script) that live alongside SKILL.md and that
the agent only reads when it actually needs them, instead of stuffing
everything into the system prompt up front. This homework asks you to write
a skill for a topic or workflow YOU pick (not sales) that bundles a SECOND
file with details the agent needs but that aren't in SKILL.md itself, then
confirm from the trace that the agent actually called `read_file` on that
second file before answering, rather than guessing.

WHAT YOU FILL IN
  TODO 1: write your own SKILL.md content. It must instruct the agent to
    read a `reference.md` file (in the same skill directory) for specific
    details it needs, and must NOT restate those details inline. The `name`
    field in your frontmatter must exactly match SKILL_NAME below.
  TODO 2: write reference.md's content: the specific facts, numbers, or
    template your skill's instructions point to and depend on.
  TODO 3: write a system prompt and a user question that should activate
    your skill.

RUN
  cd python
  uv run ./m3/m3.2_homework.py
"""

import tempfile
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend

from models import model
from my_utils import print_content

# This name becomes the skill's directory name. It must exactly match the
# `name:` field you write in the frontmatter inside build_skill_md() below.
SKILL_NAME = "story-brand"
REFERENCE_PATH = f"/skills/{SKILL_NAME}/reference.md"


# ════════════════════════════════════════════════════════════════════════
# TODO 1: Write your own SKILL.md content.
#
# Requirements:
#   - YAML frontmatter with `name` (must equal SKILL_NAME above) and
#     `description` (a specific sentence describing WHEN to use this skill).
#   - Steps that tell the agent to open `reference.md` (in this same skill
#     directory) for the specific details it needs to do the task well.
#   - Do NOT put those details in SKILL.md itself; if the agent could do
#     the task correctly without ever reading reference.md, this doesn't
#     exercise progressive disclosure.
#
# Example shape (delete this and write your own):
#   return """---
#   name: your-skill-name
#   description: Use when the user wants to ...
#   ---
#
#   # Your Skill Title
#
#   **Step 1: ...**: ...
#   **Step 2: ...**: before proceeding, read reference.md in this skill's
#     directory for the exact ... to use. Do not guess these.
#   """
# ════════════════════════════════════════════════════════════════════════

def build_skill_md() -> str:
    return """---
name: story-brand
description: Use when the user wants to generate a sales pitch for a product
---

# StoryBrand sound bites

Follow the following steps to generate the 5 sound bites for the sales pitch of a product.

**Step 1: Research**: Do a bit background research of the product that the user is trying to sell. You only need to have a high-level idea of the product, including its market and unique selling points, for a sales pitch.

**Step 2: Follow the StoryBrand framework**: Readreference.md in this skill's directory for the exact details to use. Do not guess these.

**Step 3: Generate the sound bites**: Generate the 5 sound bites based on your research at step 1 and the guidelines at step 2.
"""


# ════════════════════════════════════════════════════════════════════════
# TODO 2: Write reference.md's content.
#
# This should contain the specific facts your SKILL.md pointed to and
# depends on: a rubric, a set of numbers, a template, a checklist. Specific
# enough that an answer produced without reading it would visibly differ
# from one produced with it.
# ════════════════════════════════════════════════════════════════════════

def build_reference_md() -> str:
    return """
The StoryBrand recommends the following 5-stage P.E.A.C.E sound bites:
Problem: What problem the customer is facing.
Empathy: Show your emphasy with the customer's situation.
Answer: Propose your solution to their problem with you product.
Change: What the customer changes as a result.
End result: What the customer gains as a result.

To form a coherent story satisfying the following 4 rules:
1. make it zero cognitive load
2. link it to survival
3. make it memorable and repeatable
4. make the customer the hero

Below is an example:
Problem: :Does your dog bark incessantly when somebody knocks at the door?
Empathy: We you love your dog, but want the barking to stop.
Answer: Enroll your dog in Good Dog Academy's 1-week training program.
Change: Become the confident owner of a dog who actually listens.
End result: So you can greet people at the door with an obedient, welcoming dog at your side.
"""


# Write the skill to a scratch directory so it's discoverable through a
# FilesystemBackend, the same mechanism the lab uses for python/m3/skills/.
_tmp_root = Path(tempfile.mkdtemp(prefix="m3_2_homework_"))
_skill_dir = _tmp_root / "skills" / SKILL_NAME
_skill_dir.mkdir(parents=True, exist_ok=True)
(_skill_dir / "SKILL.md").write_text(build_skill_md())
(_skill_dir / "reference.md").write_text(build_reference_md())

backend = FilesystemBackend(root_dir=str(_tmp_root), virtual_mode=True)
print(f"Skill files written to: {_skill_dir}")


# ════════════════════════════════════════════════════════════════════════
# TODO 3: Write a system prompt and a triggering question.
#
# SYSTEM_PROMPT: give the agent a persona of your choosing (a name, a
# voice, anything you want).
# USER_QUESTION: a question that should match your skill's `description`
# closely enough that the agent activates it.
# ════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a marketing guru trying to help your user with drafting sales pitch of a product."""
USER_QUESTION = "I have a product that helps people motion graphics easily from their specifications and assets. Can you help me draft a crisp and memorable sales pitch."

agent = create_deep_agent(
    model=model,
    name="Homework_Agent",
    backend=backend,
    skills=["/skills"],
    system_prompt=SYSTEM_PROMPT,
)

result = agent.invoke({"messages": [{"role": "user", "content": USER_QUESTION}]})
print_content(model, result["messages"][-1])

read_calls = [
    call
    for msg in result["messages"]
    for call in getattr(msg, "tool_calls", [])
    if call["name"] == "read_file"
]
reference_was_read = any(call["args"].get("file_path") == REFERENCE_PATH for call in read_calls)
print(f"\n--- Did the agent read {REFERENCE_PATH}? {reference_was_read} ---")
if not reference_was_read:
    print(
        "It didn't. Either SKILL.md isn't clearly telling it to, or the "
        "task is answerable without the details in reference.md."
    )
