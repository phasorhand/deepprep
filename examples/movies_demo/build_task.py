"""Build the running example of the paper (Figure 2 / Example 1-4).

    "The goal is to construct a table for analyzing director performance across
     genres using audience ratings."

The source tables carry exactly the data-quality issues the paper's narrative
depends on:

* ``movies`` has a **duplicate row** (id 1)                     -> Deduplicate
* ``movies.title`` is inconsistently cased and padded            -> ValueTransform
* ``movies.genres`` is **multi-valued** ("Sci-Fi, Action") while
  the target schema demands single-valued categories             -> Explode
* ``ratings.values`` packs the score and the year into one field -> SplitColumn
* ``ratings`` contains years outside the target window           -> Filter
* the target needs one column per year                           -> Pivot

The ground-truth target table ``T*`` is *computed by executing the gold pipeline*
rather than hard-coded, so the fixture cannot drift away from the operators.

Run: ``python examples/movies_demo/build_task.py``
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from deepprep.operators import parse_pipeline
from deepprep.types import ADPTask, ColumnSpec, Table, TableSchema, TableSet

HERE = Path(__file__).parent

# --------------------------------------------------------------------------- #
# Source tables S
# --------------------------------------------------------------------------- #
MOVIES = pd.DataFrame(
    {
        "id": [1, 1, 17, 7, 23, 31],
        "title": [
            "  Jurassic Park ",
            "  Jurassic Park ",  # duplicate record
            "the dark knight",
            "AVATAR",
            "Inception",
            "Titanic",
        ],
        "dir_id": [101, 101, 103, 102, 103, 102],
        "genres": ["Sci-Fi", "Sci-Fi", "Sci-Fi, Action", "Sci-Fi", "Sci-Fi", "Action"],
    }
)

DIRECTORS = pd.DataFrame(
    {
        "id": [101, 102, 103],
        "name": ["Steven Spielberg", "James Cameron", "Christopher Nolan"],
    }
)

_RATING_ROWS = [
    ("jurassic park", "7.6 (2019)"),
    ("jurassic park", "8.9 (2020)"),
    ("jurassic park", "8.2 (2022)"),  # outside the target window
    ("the dark knight", "8.95 (2019)"),
    ("the dark knight", "9.2 (2020)"),
    ("avatar", "8.9 (2019)"),
    ("avatar", "8.8 (2020)"),
    ("inception", "8.8 (2019)"),
    ("inception", "8.6 (2020)"),
    ("titanic", "7.9 (2019)"),
    ("titanic", "7.7 (2020)"),
    ("titanic", "7.5 (2021)"),  # outside the target window
]
RATINGS = pd.DataFrame(_RATING_ROWS, columns=["movie", "values"])
# `rating` duplicates the score already inside `values`.  It is the decoy that
# makes the paper's failure mode reachable: SelectColumn([movie, rating]) looks
# sufficient but discards the year, and the error only surfaces several
# operators later -- which is exactly what forces a non-local revision.
RATINGS["rating"] = [float(v.split(" (")[0]) for v in RATINGS["values"]]


SOURCE_SCHEMAS = {
    "movies": TableSchema(
        description="Movies, one row per release. Contains duplicate records.",
        columns=[
            ColumnSpec("id", "int64", "Movie identifier."),
            ColumnSpec("title", "object", "Movie title; formatting is inconsistent."),
            ColumnSpec("dir_id", "int64", "Foreign key to directors.id."),
            ColumnSpec("genres", "object", "Comma-separated list of genre labels."),
        ],
    ),
    "directors": TableSchema(
        description="Directors.",
        columns=[
            ColumnSpec("id", "int64", "Director identifier."),
            ColumnSpec("name", "object", "Full name of the director."),
        ],
    ),
    "ratings": TableSchema(
        description="Audience ratings, one row per movie per year.",
        columns=[
            ColumnSpec("movie", "object", "Lower-cased movie title."),
            ColumnSpec("values", "object", "Score and year packed together, e.g. '8.2 (2022)'."),
            ColumnSpec("rating", "float64", "The score alone, without the year."),
        ],
    ),
}

# --------------------------------------------------------------------------- #
# Target schema Sigma*
# --------------------------------------------------------------------------- #
TARGET_SCHEMA = TableSchema(
    description=(
        "I need a table containing ratings by genre for each director in 2019 and 2020. "
        "One row per (director, genre) pair."
    ),
    columns=[
        ColumnSpec("director_name", "object", "The name of the director."),
        ColumnSpec("genre", "object", "A single movie genre category (never multi-valued)."),
        ColumnSpec("2019", "float64", "Average rating for the director's movies in 2019."),
        ColumnSpec("2020", "float64", "Average rating for the director's movies in 2020."),
    ],
)

# --------------------------------------------------------------------------- #
# Gold pipeline P
# --------------------------------------------------------------------------- #
GOLD_PIPELINE = """
Deduplicate(movies, [id], first)
ValueTransform(movies, title, lambda x: str(x).strip().lower())
Explode(movies, genres, sep=",")
SelectColumn(ratings, [movie, values])
SplitColumn(ratings, values, [rating_value, year], sep=" (")
ValueTransform(ratings, year, lambda x: str(x).rstrip(")"))
CastType(ratings, rating_value, float)
Filter(ratings, lambda r: r["year"] in ("2019", "2020"))
Join(movies, directors, on=(dir_id, id), how=inner, target=movies_directors_join)
Join(movies_directors_join, ratings, on=(title, movie), how=inner, target=joined)
RenameColumn(joined, {"name": "director_name", "genres": "genre"})
Pivot(joined, index=[director_name, genre], columns=[year], values=[rating_value], aggfunc=mean)
Terminate([joined])
""".strip()


def build_sources() -> TableSet:
    return TableSet(
        [
            Table("movies", MOVIES.copy(), SOURCE_SCHEMAS["movies"]),
            Table("directors", DIRECTORS.copy(), SOURCE_SCHEMAS["directors"]),
            Table("ratings", RATINGS.copy(), SOURCE_SCHEMAS["ratings"]),
        ]
    )


def build_task() -> ADPTask:
    """Assemble the task, deriving ``T*`` by executing the gold pipeline."""
    state = build_sources()
    for op in parse_pipeline(GOLD_PIPELINE):
        state = op.execute(state)
    target = state["joined"].df.copy()

    return ADPTask(
        task_id="movies_demo",
        sources=build_sources(),
        target_schema=TARGET_SCHEMA,
        target_table=target,
        gold_pipeline=[line for line in GOLD_PIPELINE.splitlines() if line.strip()],
        metadata={
            "source": "paper Figure 2 (running example)",
            "paper": "arXiv:2602.07371",
        },
    )


def main() -> None:
    task = build_task()
    out = HERE / "task.json"
    task.save(out)
    print(f"wrote {out}")
    print(f"\nTarget table T* ({task.target_table.shape[0]} rows):")
    print(task.target_table.to_string(index=False))


if __name__ == "__main__":
    main()
