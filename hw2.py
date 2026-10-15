#!/usr/bin/env python3
"""FTEC5660 HW2 student starter: build an agent that verifies CVs via MCP."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


MCP_URL = "https://ftec5660.ngrok.app/mcp"
MODEL_NAME = "deepseek-v4-flash"
THRESHOLD = 0.5


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def cv_files(folder: Path) -> list[Path]:
    """Return PDFs directly inside *folder*, sorted numerically (CV_2 before CV_10)."""

    def key(path: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else math.inf, path.name)

    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=key,
    )


def cv_text(path: Path) -> str:
    """Convert one CV PDF to markdown text."""
    from markitdown import MarkItDown

    return MarkItDown(enable_plugins=False).convert(str(path)).text_content


async def load_mcp_tools() -> list[Any]:
    """Connect to the course MCP server and return its tools as LangChain tools."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "social_graph": {
                "transport": "http",
                "url": MCP_URL,
                "headers": {"ngrok-skip-browser-warning": "true"},
            }
        }
    )
    return await client.get_tools()



def extract_score(text: str) -> float | None:
    try:
        obj = json.loads(text.strip())
        if not isinstance(obj, dict):
            return None

        value = obj.get("score")
        if isinstance(value, bool):
            return None

        score = float(value)
        if not math.isfinite(score) or not 0 <= score <= 1:
            return None

        return score
    except (ValueError, TypeError, json.JSONDecodeError):
        return None

def build_agent(tools: list[Any]) -> Any:
    from langchain_deepseek import ChatDeepSeek
    from langchain.agents import create_agent


    llm = ChatDeepSeek(model=MODEL_NAME, temperature=0.0)

    system_prompt = """You are a CV verification agent. Verify the CV against
the candidate's LinkedIn profile (primary source) and Facebook (secondary)
using the provided MCP tools.

MANDATORY WORKFLOW:
1. Extract the candidate's FULL NAME and CITY from the CV.
2. Call search_linkedin_people(q="<full name>", location="<city>").
   If 0 results → retry with fuzzy=True → try without location.
   If still 0 → call search_facebook_users(q="<full name>").
3. If multiple same-name candidates, disambiguate by city, industry,
   years_experience, and company names from the CV. Pick the BEST match.
4. Call get_linkedin_profile(person_id=<best match id>).
   Optionally cross-check with get_facebook_profile(user_id=<id>).
5. Compare CV claims with the profile.

COMPARABLE FIELDS (discrepancy only here):
  name, city, jobs (company, title, seniority, start_year, end_year),
  education (school, degree, field, end_year), skills.

TITLE / SENIORITY RULE (read carefully):
  For each job, compare the CV title with BOTH the profile's title AND
  the profile's `seniority` field.
  - CV "Senior Engineer" + profile Engineer(seniority=senior) → NOT a discrepancy.
  - CV "Senior Analyst" + profile Analyst(seniority=junior) → DISCREPANCY.
  - CV "Engineer" + profile Engineer(seniority=senior) → NOT a discrepancy.
  - Any case where CV claims HIGHER seniority than profile → DISCREPANCY.

A difference IS a discrepancy if:
  - Any start_year or end_year differs (even by 1)
  - Company or school name differs and is not an alias
  - Degree is upgraded (BSc → MSc, MSc → PhD)
  - City differs
  - CV claims a skill the profile LACKS
  - CV claims a HIGHER seniority than the profile's `seniority` field

A difference is NOT a discrepancy if:
  - Abbreviation vs full form (BSc = Bachelor of Science, HK = Hong Kong)
  - Wording/synonym (UI/UX = UI/UX Design)
  - CV lists FEWER items than profile
  - CV omits a field the profile has
  - Profile lacks the field entirely

NEVER a discrepancy: job description, headline, hometown, bio.

SCORING (be decisive):
  1.0 = all comparable fields match
  0.8 = only wording/abbreviation differences
  0.6 = profile has less info, no contradiction
  0.1 = ANY real conflict OR no matching profile found

You MUST call get_linkedin_profile at least once before scoring.

CRITICAL: Your FINAL message must contain ONLY the JSON object.
Example final message: {"score": 1.0}
"""

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent


async def score_cvs(agent: Any, cvs: dict[str, str]) -> dict[str, float | None]:
    from langchain_core.messages import HumanMessage

    print(f"[score_cvs] START, {len(cvs)} CVs", flush=True)
    sem = asyncio.Semaphore(3)
    results: dict[str, float | None] = {}

    async def verify_one(fname: str, text: str) -> tuple[str, float | None]:
        async with sem:
            try:
                resp = await agent.ainvoke(
                    {"messages": [HumanMessage(
                        content=f"Verify this CV. Output ONLY the JSON score.\n\n{text}"
                    )]},
                    config={"recursion_limit": 60},
                )
                final = resp["messages"][-1]
                score = extract_score(final.content)
                print(f"[{fname}] parsed score = {score}", flush=True)
                return fname, score if score is not None else 0.1
            except Exception as e:
                print(f"[{fname}] error: {e}", flush=True)
                return fname, 0.1

    tasks = [verify_one(f, t) for f, t in cvs.items()]
    for fname, score in await asyncio.gather(*tasks):
        results[fname] = score
    return results


# Everything below is provided runner/scoring code. No edits are needed.

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(value: Any) -> float | None:
    """Accept a float/int, or text containing exactly one number, in [0, 1]."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = str(getattr(value, "content", value))
        matches = _NUMBER_RE.findall(text)
        if len(matches) != 1:
            return None
        score = float(matches[0])
    if math.isnan(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read labels (1 = valid CV, 0 = has discrepancy) and reasons from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return {
        name: entry if isinstance(entry, dict) else {"label": entry}
        for name, entry in json.loads(path.read_text(encoding="utf-8")).items()
    }


def correctness_text(score: float | None, expected: dict[str, Any] | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if score is None:
        return "incorrect: score is missing or not a number in [0, 1]"
    if expected is None:
        return "not graded: no ground truth for this CV"
    label = int(expected["label"])
    predicted = 1 if score > THRESHOLD else 0
    if predicted == label:
        return "correct"
    reason = f" ({expected['reason']})" if expected.get("reason") else ""
    return f"incorrect: expected {label}{reason}, predicted {predicted}"


def write_results(names: list[str], scores: dict[str, Any], truth: dict[str, dict[str, Any]]) -> tuple[Path, int]:
    """Write the required results.csv file and return how many CVs were correct."""
    output = Path("results.csv")
    correct = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cv", "score", "correctness"])
        for name in names:
            score = parse_score(scores.get(name))
            verdict = correctness_text(score, truth.get(name))
            correct += verdict == "correct"
            writer.writerow([name, "" if score is None else f"{score:.4f}", verdict])
    return output, correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW2 on CV PDFs")
    parser.add_argument(
        "--cv-folder",
        required=True,
        type=Path,
        help="folder containing CV PDF files",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    paths = cv_files(folder)
    if not paths:
        raise SystemExit(f"no PDF files found in {folder}")

    load_env_file()
    cvs = {path.name: cv_text(path) for path in paths}
    tools = await load_mcp_tools()
    agent = build_agent(tools)
    scores = await score_cvs(agent, cvs)
    if not isinstance(scores, dict):
        raise TypeError("score_cvs() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct = write_results(list(cvs), scores, truth)
    summary = f" Accuracy: {correct}/{len(cvs)}." if truth else ""
    print(f"Processed {len(cvs)} CV(s). Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.cv_folder.is_dir():
        raise SystemExit(f"not a folder: {args.cv_folder}")
    return asyncio.run(run(args.cv_folder))


if __name__ == "__main__":
    raise SystemExit(main())
