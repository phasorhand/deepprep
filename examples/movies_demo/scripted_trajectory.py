"""A canned agent trajectory reproducing Figure 4 / Example 4 of the paper.

Driving :class:`~deepprep.agent.llm.ScriptedClient` with these responses exercises
the complete tree-based reasoning mechanism with no network access:

  1. **Initialization** — expand from the root ``n0``; ``Deduplicate``,
     ``ValueTransform`` and ``SelectColumn`` produce ``n1, n2, n3``.
  2. **Execution failure** — an ``ExeCode`` expansion from ``n3`` raises a
     missing-column error, recorded as the failure node ``n4`` with no state.
  3. **Backtracking and alternative expansion** — the plan attributes the error
     to the *earlier* ``SelectColumn`` decision, points ``<from>`` at ``n2``, and
     expands a corrected branch that keeps the ``values`` column.
  4. Continue on the corrected branch to the target schema.
  t. **Termination** — ``<answer>`` returns the root-to-leaf pipeline ``P*``.

Step 3 is the whole point of the paper: a linear ReAct agent cannot make it,
because ``n2`` is no longer reachable once ``n3`` has been committed to.
"""

from __future__ import annotations

#: Root -> n2 prefix, reused verbatim by the backtracking turn.
_PREFIX_N2 = """Deduplicate(movies, [id], first)
ValueTransform(movies, title, lambda x: str(x).strip().lower())"""

#: Root -> n9 prefix (the corrected branch).
_PREFIX_N9 = (
    _PREFIX_N2
    + """
SelectColumn(ratings, [movie, values])
SplitColumn(ratings, values, [rating_value, year], sep=" (")
ValueTransform(ratings, year, lambda x: str(x).rstrip(")"))
CastType(ratings, rating_value, float)
Filter(ratings, lambda r: r["year"] in ("2019", "2020"))"""
)

_FINAL_PIPELINE = (
    _PREFIX_N9
    + """
Explode(movies, genres, sep=",")
Join(movies, directors, on=(dir_id, id), how=inner, target=movies_directors_join)
Join(movies_directors_join, ratings, on=(title, movie), how=inner, target=joined)
RenameColumn(joined, {"name": "director_name", "genres": "genre"})
Pivot(joined, index=[director_name, genre], columns=[year], values=[rating_value], aggfunc=mean)
Terminate([joined])"""
)


TURN_1_INIT = """
<plan>
The tree contains only the root n0. Comparing the source tables with the target
schema: `movies` has a duplicate record for id 1, and its titles are padded and
inconsistently cased while `ratings.movie` is lower-cased, so a title join would
match nothing. I will clean `movies` from n0 and narrow `ratings` to the columns
I expect to need.
</plan>
<expand>
<from>
</from>
<ops>
Deduplicate(movies, [id], first)
ValueTransform(movies, title, lambda x: str(x).strip().lower())
SelectColumn(ratings, [movie, rating])
</ops>
</expand>
""".strip()


TURN_2_FAILURE = f"""
<plan>
n3 holds the cleaned tables. The target needs one column per year, so I now
integrate the rating values into `movies` and pull the year out alongside them.
</plan>
<expand>
<from>
{_PREFIX_N2}
SelectColumn(ratings, [movie, rating])
</from>
<ops>
ExeCode([movies, ratings], movies_with_ratings, ```
merged = movies.merge(ratings, left_on='title', right_on='movie')
merged['year'] = ratings['year']
movies_with_ratings = merged
```)
</ops>
</expand>
""".strip()


TURN_3_BACKTRACK = f"""
<plan>
The error is not local to the ExeCode operator. `ratings` has no `year` column
because the year was only ever encoded inside `values`, and my SelectColumn at n3
dropped `values`. The faulty decision therefore sits one step upstream, so I
backtrack to n2 -- the state before SelectColumn -- and expand an alternative
branch that keeps `values`, splits the score and the year out of it, and filters
to the two years the target schema asks for. The prefix n0->n2 stays valid and is
reused.
</plan>
<expand>
<from>
{_PREFIX_N2}
</from>
<ops>
SelectColumn(ratings, [movie, values])
SplitColumn(ratings, values, [rating_value, year], sep=" (")
ValueTransform(ratings, year, lambda x: str(x).rstrip(")"))
CastType(ratings, rating_value, float)
Filter(ratings, lambda r: r["year"] in ("2019", "2020"))
</ops>
</expand>
""".strip()


TURN_4_COMPLETE = f"""
<plan>
The corrected branch gives `ratings(movie, rating_value, year)` restricted to
2019 and 2020. What remains is the target's structure: `genres` is still
multi-valued ("Sci-Fi, Action") whereas the target demands a single category per
row, so I explode it; then I attach director names, join on the normalized title,
rename to the target column names, and pivot the years into columns.
</plan>
<expand>
<from>
{_PREFIX_N9}
</from>
<ops>
Explode(movies, genres, sep=",")
Join(movies, directors, on=(dir_id, id), how=inner, target=movies_directors_join)
Join(movies_directors_join, ratings, on=(title, movie), how=inner, target=joined)
RenameColumn(joined, {{"name": "director_name", "genres": "genre"}})
Pivot(joined, index=[director_name, genre], columns=[year], values=[rating_value], aggfunc=mean)
</ops>
</expand>
""".strip()


TURN_5_ANSWER = f"""
<plan>
The leaf state's `joined` table has exactly the target columns -- director_name,
genre, 2019, 2020 -- with one row per (director, single genre) pair and no
multi-valued categories. The target schema is satisfied, so I extract the
root-to-leaf pipeline and terminate.
</plan>
<answer>
{_FINAL_PIPELINE}
</answer>
""".strip()


#: The full scripted run, in order.
FIGURE_4_TRAJECTORY: list[str] = [
    TURN_1_INIT,
    TURN_2_FAILURE,
    TURN_3_BACKTRACK,
    TURN_4_COMPLETE,
    TURN_5_ANSWER,
]
