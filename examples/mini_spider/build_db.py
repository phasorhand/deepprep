"""Build a miniature Spider-layout NL2SQL benchmark.

The Sec 5.3 synthesis pipeline turns (database, question, SQL) triples into ADP
tasks, so exercising it end to end needs real SQLite files and a real benchmark
spec.  Spider itself is a 1.3 GB download; this fixture reproduces its *shape*
-- ``database/<db_id>/<db_id>.sqlite`` plus a JSON list of
``{db_id, question, query}`` -- at a size that runs in a second.

    python examples/mini_spider/build_db.py [out_dir | --out out_dir]

Two databases with multi-table joins, aggregation and grouping, chosen so the
reversibility gate of Sec 5.3 has something to reject as well as accept.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# schema + rows
# --------------------------------------------------------------------------- #
UNIVERSITY = {
    "department": (
        "CREATE TABLE department (dept_id INTEGER PRIMARY KEY, dept_name TEXT, "
        "building TEXT, budget REAL)",
        [
            (1, "Computer Science", "Turing Hall", 1850000.0),
            (2, "Mathematics", "Euler Hall", 920000.0),
            (3, "Physics", "Bohr Hall", 1340000.0),
            (4, "History", "Herodotus Hall", 410000.0),
        ],
    ),
    "instructor": (
        "CREATE TABLE instructor (inst_id INTEGER PRIMARY KEY, name TEXT, "
        "dept_id INTEGER, salary REAL, hired_year INTEGER)",
        [
            (1, "Ada Lovelace", 1, 145000.0, 2011),
            (2, "Alan Turing", 1, 162000.0, 2008),
            (3, "Emmy Noether", 2, 138000.0, 2013),
            (4, "Carl Gauss", 2, 151000.0, 2005),
            (5, "Marie Curie", 3, 158000.0, 2009),
            (6, "Niels Bohr", 3, 149000.0, 2015),
            (7, "Herodotus Jones", 4, 96000.0, 2018),
        ],
    ),
    "course": (
        "CREATE TABLE course (course_id TEXT PRIMARY KEY, title TEXT, "
        "dept_id INTEGER, credits INTEGER)",
        [
            ("CS101", "Intro to Computing", 1, 4),
            ("CS220", "Algorithms", 1, 4),
            ("CS330", "Databases", 1, 3),
            ("MA101", "Calculus I", 2, 4),
            ("MA210", "Linear Algebra", 2, 3),
            ("PH150", "Classical Mechanics", 3, 4),
            ("PH260", "Quantum Mechanics", 3, 3),
            ("HI110", "Ancient Worlds", 4, 3),
        ],
    ),
    "enrollment": (
        "CREATE TABLE enrollment (enroll_id INTEGER PRIMARY KEY, course_id TEXT, "
        "student_name TEXT, semester TEXT, grade REAL)",
        [
            (1, "CS101", "Rui Chen", "2023-Fall", 3.7),
            (2, "CS101", "Mia Sato", "2023-Fall", 3.9),
            (3, "CS101", "Omar Aziz", "2024-Spring", 3.1),
            (4, "CS220", "Rui Chen", "2024-Spring", 3.5),
            (5, "CS220", "Lena Ortiz", "2024-Spring", 4.0),
            (6, "CS330", "Mia Sato", "2024-Spring", 3.8),
            (7, "MA101", "Omar Aziz", "2023-Fall", 2.9),
            (8, "MA101", "Lena Ortiz", "2023-Fall", 3.6),
            (9, "MA210", "Rui Chen", "2024-Spring", 3.3),
            (10, "PH150", "Mia Sato", "2023-Fall", 3.4),
            (11, "PH260", "Lena Ortiz", "2024-Spring", 3.95),
            (12, "HI110", "Omar Aziz", "2024-Spring", 3.2),
        ],
    ),
}

FLIGHTS = {
    "airline": (
        "CREATE TABLE airline (airline_id INTEGER PRIMARY KEY, airline_name TEXT, "
        "country TEXT, founded INTEGER)",
        [
            (1, "Blue Sky", "USA", 1994),
            (2, "Alpine Air", "Switzerland", 1981),
            (3, "Pacific Wing", "Japan", 2003),
        ],
    ),
    "airport": (
        "CREATE TABLE airport (airport_code TEXT PRIMARY KEY, city TEXT, country TEXT)",
        [
            ("SFO", "San Francisco", "USA"),
            ("JFK", "New York", "USA"),
            ("ZRH", "Zurich", "Switzerland"),
            ("NRT", "Tokyo", "Japan"),
        ],
    ),
    "flight": (
        "CREATE TABLE flight (flight_id INTEGER PRIMARY KEY, airline_id INTEGER, "
        "origin TEXT, destination TEXT, distance_km INTEGER, delay_minutes INTEGER)",
        [
            (1, 1, "SFO", "JFK", 4130, 12),
            (2, 1, "JFK", "SFO", 4130, 0),
            (3, 1, "SFO", "NRT", 8280, 35),
            (4, 2, "ZRH", "JFK", 6330, 8),
            (5, 2, "ZRH", "NRT", 9600, 22),
            (6, 2, "JFK", "ZRH", 6330, 5),
            (7, 3, "NRT", "SFO", 8280, 41),
            (8, 3, "NRT", "ZRH", 9600, 17),
            (9, 3, "NRT", "JFK", 10850, 63),
        ],
    ),
}

DATABASES = {"university": UNIVERSITY, "flights": FLIGHTS}

# --------------------------------------------------------------------------- #
# questions
# --------------------------------------------------------------------------- #
SPEC = [
    {
        "db_id": "university",
        "question": "What is the name and salary of every instructor, along with the "
                    "department they belong to?",
        "query": "SELECT i.name, d.dept_name, i.salary FROM instructor i "
                 "JOIN department d ON i.dept_id = d.dept_id",
    },
    {
        "db_id": "university",
        "question": "For each department, how many instructors are there and what is "
                    "their average salary?",
        "query": "SELECT d.dept_name, COUNT(*) AS n_instructors, AVG(i.salary) AS avg_salary "
                 "FROM instructor i JOIN department d ON i.dept_id = d.dept_id "
                 "GROUP BY d.dept_name",
    },
    {
        "db_id": "university",
        "question": "List the title and department name of every course worth 4 credits.",
        "query": "SELECT c.title, d.dept_name FROM course c "
                 "JOIN department d ON c.dept_id = d.dept_id WHERE c.credits = 4",
    },
    {
        "db_id": "university",
        "question": "What is the average grade for each course title?",
        "query": "SELECT c.title, AVG(e.grade) AS avg_grade FROM enrollment e "
                 "JOIN course c ON e.course_id = c.course_id GROUP BY c.title",
    },
    {
        "db_id": "university",
        "question": "Which departments have a budget over one million, and what building "
                    "are they in?",
        "query": "SELECT dept_name, building, budget FROM department WHERE budget > 1000000",
    },
    {
        "db_id": "flights",
        "question": "Show each flight's airline name, origin city and destination city.",
        "query": "SELECT a.airline_name, o.city AS origin_city, dst.city AS destination_city "
                 "FROM flight f JOIN airline a ON f.airline_id = a.airline_id "
                 "JOIN airport o ON f.origin = o.airport_code "
                 "JOIN airport dst ON f.destination = dst.airport_code",
    },
    {
        "db_id": "flights",
        "question": "For each airline, what is the total distance flown and the average delay?",
        "query": "SELECT a.airline_name, SUM(f.distance_km) AS total_km, "
                 "AVG(f.delay_minutes) AS avg_delay FROM flight f "
                 "JOIN airline a ON f.airline_id = a.airline_id GROUP BY a.airline_name",
    },
    {
        "db_id": "flights",
        "question": "List the flights longer than 8000 km with their airline and delay.",
        "query": "SELECT a.airline_name, f.origin, f.destination, f.distance_km, "
                 "f.delay_minutes FROM flight f JOIN airline a ON f.airline_id = a.airline_id "
                 "WHERE f.distance_km > 8000",
    },
    {
        "db_id": "flights",
        "question": "How many flights depart from each airport city?",
        "query": "SELECT o.city, COUNT(*) AS n_departures FROM flight f "
                 "JOIN airport o ON f.origin = o.airport_code GROUP BY o.city",
    },
]


def build(out_dir: Path) -> Path:
    db_root = out_dir / "database"
    db_root.mkdir(parents=True, exist_ok=True)

    for db_id, tables in DATABASES.items():
        d = db_root / db_id
        d.mkdir(exist_ok=True)
        path = d / f"{db_id}.sqlite"
        path.unlink(missing_ok=True)
        conn = sqlite3.connect(path)
        for name, (ddl, rows) in tables.items():
            conn.execute(ddl)
            placeholders = ",".join("?" * len(rows[0]))
            conn.executemany(f"INSERT INTO {name} VALUES ({placeholders})", rows)
        conn.commit()
        conn.close()
        print(f"wrote {path} ({len(tables)} tables)")

    spec_path = out_dir / "spec.json"
    spec_path.write_text(json.dumps(SPEC, indent=2))
    print(f"wrote {spec_path} ({len(SPEC)} cases)")
    return spec_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Accepted both ways: the README documents the positional form, but `--out`
    # is what everyone reaches for, and taking argv[1] on faith turned a
    # mistyped flag into a directory *named* `--out` while the intended
    # destination stayed empty.
    ap.add_argument("out_dir", nargs="?", default=None, type=Path)
    ap.add_argument("--out", dest="out_flag", default=None, type=Path)
    args = ap.parse_args(argv)

    if args.out_dir is not None and args.out_flag is not None:
        ap.error("give the output directory once, positionally or with --out")
    build(args.out_dir or args.out_flag or HERE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
