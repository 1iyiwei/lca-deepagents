# python/m4/m4.3_homework.py
"""M4.3 Homework: Write Your Own Dynamic Subagent Workflow.

THE IDEA
The lab gave the main agent a 2MB manuscript split into labeled books and
had it write a "workflow" that dispatched one book-scanner subagent per
book, so the full corpus never entered the main model's own context. This
homework asks you to do the same shape of thing on a scenario of your own
choosing: a synthetic corpus of your own, split into your own labeled
sections, and a subagent that scans each section for something other than
anachronisms.

A few starting points, if you want one:
  - A Sherlock Holmes story, split by chapter, scanned for clues the
    detective mentions but never actually explains.
  - The script of Bee Movie or Shrek, split by scene, scanned for lines
    that don't match the character who supposedly says them.
  - Your own corrupted classic, like the lab's, but seeded with a
    different kind of error: wrong units, swapped character names,
    continuity errors between chapters.

WHAT YOU FILL IN
  TODO 1: write your own corpus, a single string split into at least 5
    labeled sections using a consistent header format (like the lab's
    "=== EPIC BOOK N ===").
  TODO 2: write the section-scanner's system prompt (what should it flag in
    one section?) and the main agent's system prompt (telling it to run a
    workflow that splits your corpus and dispatches one scanner call per
    section).

RUN
  cd python
  uv run ./m4/m4.3_homework.py

NOTE
  This uses the code interpreter (langchain_quickjs), same as the lab. Make
  sure you've run `uv sync` from python/ so it's installed.
"""

from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_quickjs import CodeInterpreterMiddleware

from models import model, strong_model
from my_utils import print_content

DATA_DIR = Path(__file__).resolve().parent / "homework_data"
DATA_DIR.mkdir(exist_ok=True)
CORPUS_PATH = DATA_DIR / "my_corpus.txt"


# ════════════════════════════════════════════════════════════════════════
# TODO 1: Write your own corpus.
#
# Requirements:
#   - A single string with at least 5 labeled sections.
#   - Pick a consistent header format, e.g. "=== SECTION N ===" or
#     "=== TICKET N ===", and stick to it exactly: the main agent's prompt
#     (TODO 2) needs to describe the same format so it can split on it.
#   - Plant something worth finding in a few of the sections (an off-topic
#     sentence, a specific keyword, whatever your scanner in TODO 2 is
#     looking for) so there's something for the workflow to actually
#     surface.
#
# Example shape (delete this and write your own):
#   return """\
#   === SECTION 1 ===
#   ...
#
#   === SECTION 2 ===
#   ...
#   """
# ════════════════════════════════════════════════════════════════════════

def build_corpus() -> str:
    return """\
    The book of the horror.
    Each section contains a paragraph of horror story,
    but some mixed with one or two funny sentences.

=== SECTION 1 ===
The old mansion loomed against the moonlit sky. Shadows crept along the crumbling walls. Thunder rumbled ominously in the distance. The windows stared like hollow eyes from a decaying face. I wonder if they serve pizza in the mansion's kitchen. The air was thick with dread and the musty smell of centuries past.

=== SECTION 2 ===
I pushed open the creaking door and stepped inside. Dust particles danced in the beam of my flashlight. Strange paintings lined the hallway, their eyes following my every move. A sudden chill ran down my spine as I heard a faint whisper. The ghost told me a really funny joke about skeletons. The portraits seemed to come alive in the flickering shadows.

=== SECTION 3 ===
The basement stairs descended into complete darkness. Each step groaned beneath my weight like a dying animal. Cobwebs brushed against my face as I descended. A metallic smell filled my nostrils. Did you know that bananas are berries? Chains hung from the walls, and scratches marred the stone floor below them.

=== SECTION 4 ===
The ritual chamber was exactly as the legends described. Candles burned with an eerie blue flame around the pentagram. Ancient symbols covered every surface. The air seemed to pulse with malevolent energy. I really should remember to water my plants at home. A book lay open on the altar, its pages filled with incomprehensible text that seemed to writhe before my eyes.

=== SECTION 5 ===
I ran from the mansion without looking back. My heart pounded as I heard footsteps chasing me through the halls. The doors slammed open behind me as if propelled by invisible hands. I stumbled through the front entrance and into the night. That was the best three-legged race I've ever participated in! Never again would I return to that forsaken place, and I have never slept soundly since that terrifying night.
    """


CORPUS_PATH.write_text(build_corpus())


# ════════════════════════════════════════════════════════════════════════
# TODO 2: Write the scanner and main agent prompts.
#
# Return (scanner_prompt, main_prompt):
#   - scanner_prompt: what the section-scanner subagent should look for in
#     ONE section it's handed, and what it should return.
#   - main_prompt: tells the main agent about the corpus file, the header
#     format from TODO 1, and to run a WORKFLOW that splits the corpus and
#     dispatches one scanner call per section (the word "workflow" is what
#     triggers code-based dispatch, see the lesson).
# ════════════════════════════════════════════════════════════════════════

def build_prompts() -> tuple[str, str]:
    SCANNER_PROMPT="""
You are a section scanner. Your job is to scan one section of this book that is supposed to be scary, and find any sentences that are actually funny or not scary.
If you find any such sentences, return them in a list.
Otherwise, return an empty list.
There might not be any non-scary sentence in a section, so do not force it.
"""
    MAIN_PROMPT="""
You are a book scanner. Your job is to scan an entire book for funny or non-scary sentences.
Run a workflow that splits the book into multiple sections based on the section dividers assigns each section to a subagent called "section-scanner", collects the results from each section scanner, and returns the combined results.
"""
    return SCANNER_PROMPT, MAIN_PROMPT

SCANNER_PROMPT, MAIN_PROMPT = build_prompts()

section_scanner = {
    "name": "section-scanner",
    "description": (
        "Scan one section of the corpus for whatever the scanner prompt "
        "asks for. Delegate one section per call."
    ),
    "system_prompt": SCANNER_PROMPT,
    "model": model,
}

agent = create_deep_agent(
    model=strong_model,
    middleware=[CodeInterpreterMiddleware()],
    system_prompt=MAIN_PROMPT,
    subagents=[section_scanner],
    backend=FilesystemBackend(root_dir=DATA_DIR, virtual_mode=True),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Run a workflow to scan every section of my_corpus.txt and report what you find.",
            }
        ]
    },
    config={"recursion_limit": 100},
)

print_content(model, result["messages"][-1])
