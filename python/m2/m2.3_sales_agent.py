from pathlib import Path

from deepagents import create_deep_agent

from local_sandbox_backend import create_backend, sandbox_path
from models import model

DB_PATH = Path(__file__).resolve().parent / "chinook.db"
DB_SANDBOX_PATH = sandbox_path("chinook.db")
CHART_SANDBOX_PATH = sandbox_path("genre_revenue.png")

backend, cleanup = create_backend("lca-deepagents-lab")

with open(DB_PATH, "rb") as f:
    upload_results = backend.upload_files([(DB_SANDBOX_PATH, f.read())])

for upload_result in upload_results:
    if upload_result.error:
        raise RuntimeError(f"Failed to upload {upload_result.path}: {upload_result.error}")

agent = create_deep_agent(
    model=model,
    backend=backend,
    system_prompt=(
        f"You are a sales data analyst with access to the Chinook music store database "
        f"at {DB_SANDBOX_PATH}. Use sqlite3 and matplotlib to answer questions with charts. "
        f"Install any packages you need with pip before importing them. "
        f"When asked to produce a chart, write a Python script, execute it, and confirm "
        f"the output file was created."
    ),
)

try:
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Query the Chinook database at {DB_SANDBOX_PATH} to get total revenue "
                        "by genre. Create a clean donut chart showing each genre's share "
                        "of total sales revenue. Group any genres that individually "
                        "account for less than 3% of total revenue into a single 'Other' "
                        "slice. Label each slice with the genre name and percentage. "
                        "Use a visually distinct color palette, leave a white center hole, "
                        "and make sure no labels overlap with each other or with the title. "
                        "Add enough top padding so the title is fully visible. "
                        f"Save the chart to {CHART_SANDBOX_PATH}."
                    ),
                }
            ]
        }
    )
    print(result["messages"][-1].content)

    [download] = backend.download_files([CHART_SANDBOX_PATH])
    if download.error:
        raise RuntimeError(f"Failed to download {download.path}: {download.error}")
    out_path = Path(__file__).parent / "genre_revenue.png"
    out_path.write_bytes(download.content)
    print(f"Chart saved to {out_path}")

finally:
    cleanup()
