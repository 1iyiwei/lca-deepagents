# python/m2/m2.3_homework_filled.py
"""Reference copy of m2.3_homework.py with TODOs 1-3 filled in so you
can run it end to end and see what "done" looks like. This is just one
possible answer, so yours might be different. Explore!"""

from pathlib import Path

from deepagents import create_deep_agent

from local_sandbox_backend import create_backend, sandbox_path
from models import model

# TODO 1 filled in
SYSTEM_PROMPT = (
    "You are a data visualization assistant. When asked to run code, "
    "write the script to a file first, then execute it. Install any "
    "packages you need with pip before importing them. When asked for "
    "a chart, use matplotlib and save it as a .png file."
)

# TODO 3 filled in (moved up: TASK_TWO references it)
CHART_PATH = sandbox_path("chart.png")

# TODO 2 filled in
TASK_ONE = (
    "Generate 12 months of made-up monthly rainfall totals (in mm) for "
    "a fictional city, save them to rainfall.json, and print them."
)
TASK_TWO = (
    "Read rainfall.json (don't regenerate the numbers) and create a bar "
    f"chart of monthly rainfall. Save it to {CHART_PATH}."
)

backend, cleanup = create_backend("lca-deepagents-homework")

agent = create_deep_agent(
    model=model,
    backend=backend,
    system_prompt=SYSTEM_PROMPT,
)

try:
    result = agent.invoke({"messages": [{"role": "user", "content": TASK_ONE}]})
    print("--- Task 1 ---")
    print(result["messages"][-1].content)

    result = agent.invoke({"messages": [{"role": "user", "content": TASK_TWO}]})
    print("\n--- Task 2 (same sandbox, should see Task 1's file) ---")
    print(result["messages"][-1].content)

    [download] = backend.download_files([CHART_PATH])
    if download.error:
        raise RuntimeError(f"Failed to download {CHART_PATH}: {download.error}")
    out_path = Path(__file__).parent / "homework_chart.png"
    out_path.write_bytes(download.content)
    print(f"Chart saved to {out_path}")
finally:
    cleanup()
