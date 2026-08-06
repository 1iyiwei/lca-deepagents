# python/m2/m2.4_homework.py
"""M2.4 Homework: Ask Your Own Question, Then Grade It.

THE IDEA
Lab 2 asked one fixed, dependent-query question about the Chinook
database and let the interpreter chain four SQL queries together inside
a single eval() call. This homework asks you to pose your OWN question
about the database (anything query_chinook and a little JavaScript can
answer, simple or dependent, your call), and then, since this lesson is
also about evaluating what an agent's code produces, write your own quick
eval check that judges whether the agent's answer looks right. There's no
single right question or eval method here, that's the point.

WHAT YOU FILL IN
  TODO 1: write your own natural-language question about the Chinook
    database (see the schema hints in SYSTEM below) for the interpreter
    agent to answer using eval() and query_chinook.
  TODO 2: write a small eval_answer(...) function that independently
    checks whether the agent's final answer looks correct. However you
    want to do this is fine: run your own SQL query and compare, check
    for an expected keyword or number, or just print both side by side
    for your own judgment call. Pick whatever level of rigor makes sense
    for your question.

RUN
  cd python
  uv run ./m2/m2.4_homework.py
"""

import json
import sqlite3
import uuid
from pathlib import Path

from deepagents import create_deep_agent
from langchain.tools import tool
from langchain_quickjs import CodeInterpreterMiddleware

from models import model

DB_PATH = Path(__file__).resolve().parent / "chinook.db"

SYSTEM = (
    "You are a sales analyst for Chinook Digital Music Store. "
    "Use the query_chinook tool to query the database. "
    "Key tables: Artist(ArtistId, Name), Album(AlbumId, Title, ArtistId), "
    "Track(TrackId, Name, AlbumId, GenreId), Genre(GenreId, Name), "
    "Customer(CustomerId, FirstName, LastName, Country), "
    "Invoice(InvoiceId, CustomerId), "
    "InvoiceLine(InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity). "
    "Revenue is InvoiceLine.UnitPrice * InvoiceLine.Quantity. "
    "The eval tool supports Programmatic Tool Calling (PTC): JavaScript "
    "running inside eval() can call query_chinook via tools.queryChinook()."
)

MIDDLEWARE = (
        " The eval tool supports Programmatic Tool Calling (PTC): JavaScript"
        " running inside eval() can call query_chinook via tools.queryChinook()."
        " For dependent queries where each answer requires a result from the"
        " previous, prefer a single eval() call that chains all queries in"
        " JavaScript — intermediate values stay in variables and never return to the model."
)

@tool
def query_chinook(sql: str) -> str:
    """Execute a read-only SQL query against the Chinook database. Returns a JSON-encoded string."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql)
        rows = [dict(row) for row in cursor.fetchall()]
        return json.dumps(rows)
    finally:
        conn.close()


# ════════════════════════════════════════════════════════════════════════
# TODO 1: Write your own question about the Chinook database.
#
# Pick anything the query_chinook tool can answer: top customers by
# country, which artist has the most albums, average invoice total by
# year, whatever you're curious about. A question with a couple of
# dependent steps (like Lab 2's) is a good excuse to use PTC, but a
# single-query question is a perfectly fine answer too.
# ════════════════════════════════════════════════════════════════════════

TASK = """
Query the Chinook database to find out (1) which artist has the highest total revenues, (2) which album of that artist has the highest revenue, and (3) which track on that album has the highest revenue.
Use the query_chinook tool to answer these questions.
"""


# ════════════════════════════════════════════════════════════════════════
# TODO 2: Write a simple eval check for the agent's answer.
#
# eval_answer(answer_text) runs after the agent responds. Independently
# work out what you believe the right answer is (run your own SQL query,
# do the math by hand, whatever) and compare it against answer_text. Print
# whatever verdict makes sense; this doesn't need to be a strict
# pass/fail, a reasoned printout is fine.
# ════════════════════════════════════════════════════════════════════════


def _top_artist_by_revenue(conn: sqlite3.Connection) -> sqlite3.Row:
    return conn.execute(
        """
        SELECT Artist.ArtistId, Artist.Name,
               SUM(InvoiceLine.UnitPrice * InvoiceLine.Quantity) AS rev
        FROM InvoiceLine
        JOIN Track USING(TrackId)
        JOIN Album USING(AlbumId)
        JOIN Artist USING(ArtistId)
        GROUP BY Artist.ArtistId
        ORDER BY rev DESC
        LIMIT 1
        """
    ).fetchone()


def _top_album_by_revenue(conn: sqlite3.Connection, artist_id: int) -> sqlite3.Row:
    return conn.execute(
        """
        SELECT Album.AlbumId, Album.Title,
               SUM(InvoiceLine.UnitPrice * InvoiceLine.Quantity) AS rev
        FROM InvoiceLine
        JOIN Track USING(TrackId)
        JOIN Album USING(AlbumId)
        WHERE Album.ArtistId = ?
        GROUP BY Album.AlbumId
        ORDER BY rev DESC
        LIMIT 1
        """,
        (artist_id,),
    ).fetchone()


def _top_track_by_revenue(conn: sqlite3.Connection, album_id: int) -> sqlite3.Row:
    return conn.execute(
        """
        SELECT Track.TrackId, Track.Name,
               SUM(InvoiceLine.UnitPrice * InvoiceLine.Quantity) AS rev
        FROM InvoiceLine
        JOIN Track USING(TrackId)
        WHERE Track.AlbumId = ?
        GROUP BY Track.TrackId
        ORDER BY rev DESC
        LIMIT 1
        """,
        (album_id,),
    ).fetchone()


def _expected_answer() -> tuple[sqlite3.Row, sqlite3.Row, sqlite3.Row]:
    """Independently derive the expected artist/album/track chain via direct SQL."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        top_artist = _top_artist_by_revenue(conn)
        top_album = _top_album_by_revenue(conn, top_artist["ArtistId"])
        top_track = _top_track_by_revenue(conn, top_album["AlbumId"])
        return top_artist, top_album, top_track
    finally:
        conn.close()


def eval_answer(answer_text: str) -> None:
    """Independently compute the expected artist/album/track chain, then
    check whether all three show up in the agent's answer."""
    top_artist, top_album, top_track = _expected_answer()

    print("\n--- Eval check ---")
    print(f"Expected top artist: {top_artist['Name']} (${top_artist['rev']:.2f})")
    print(f"Expected top album: {top_album['Title']} (${top_album['rev']:.2f})")
    print(f"Expected top track: {top_track['Name']} (${top_track['rev']:.2f})")

    checks = {
        "mentions expected artist": top_artist["Name"].lower() in answer_text.lower(),
        "mentions expected album": top_album["Title"].lower() in answer_text.lower(),
        "mentions expected track": top_track["Name"].lower() in answer_text.lower(),
    }
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")


if TASK is None:
    raise NotImplementedError("TODO 1: see the comment block above")

agent = create_deep_agent(
    model=model,
    tools=[query_chinook],
    middleware=[CodeInterpreterMiddleware(ptc=["query_chinook"])],
    system_prompt=SYSTEM+MIDDLEWARE,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": TASK}]},
    config={"configurable": {"thread_id": str(uuid.uuid4())}},
)

answer = result["messages"][-1].content
print(answer)
eval_answer(answer)
