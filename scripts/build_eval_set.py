#!/usr/bin/env python3
"""Validate golden-draft.json against the real repo and emit evals/golden.json.

Checks, in order:
  1. every expected_doc_path exists in the cloned Adobe repo
  2. the citation mapper reproduces every expected_url
  3. reports anything it had to correct

Also embeds the TOC.md text for each guide the eval set touches, so the unit
suite can assert URL mapping without a 2.7 GB checkout present. That keeps CI
hermetic.

Usage:  python scripts/build_eval_set.py [--repo /tmp/adobedocs]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onramp.citations import build_url, parse_toc, product_area  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAFT = REPO_ROOT / "golden-draft.json"
OUT = REPO_ROOT / "evals" / "golden.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/tmp/adobedocs", help="path to the cloned Adobe docs repo")
    args = ap.parse_args()

    repo = Path(args.repo)
    if not (repo / "help").is_dir():
        print(f"ERROR: {repo}/help not found.")
        print("  Check: did the clone finish? Re-run with")
        print("  git clone --depth 1 https://github.com/AdobeDocs/experience-platform.en.git /tmp/adobedocs")
        return 1

    if not DRAFT.exists():
        print(f"ERROR: {DRAFT} not found. Check that golden-draft.json is in the repo root.")
        return 1

    data = json.loads(DRAFT.read_text())
    questions = data["questions"]

    # --- 1. paths exist ----------------------------------------------------
    missing = [q["id"] for q in questions if not (repo / q["expected_doc_path"]).is_file()]
    if missing:
        print(f"ERROR: {len(missing)} expected_doc_path values do not exist: {missing}")
        print("  Check: is the clone current? These paths may have been renamed upstream.")
        return 1
    print(f"paths       : {len(questions)}/{len(questions)} exist in {repo}")

    # --- 2. collect the TOCs the eval set needs ---------------------------
    guides = sorted({product_area(q["expected_doc_path"]) for q in questions})
    fixtures: dict[str, str] = {}
    indexes = {}
    for guide in guides:
        toc = repo / "help" / guide / "TOC.md"
        if toc.exists():
            text = toc.read_text(encoding="utf-8", errors="replace")
            fixtures[guide] = text
            indexes[guide] = parse_toc(text, guide)
        else:
            # Not every guide ships a TOC.md; those fall back to the naive rule.
            print(f"note        : no TOC.md for {guide!r}, naive mapping will be used")

    # --- 3. URL mapping ----------------------------------------------------
    corrected = []
    failures = []
    for q in questions:
        path = q["expected_doc_path"]
        got = build_url(path, indexes.get(product_area(path)))
        if got != q["expected_url"]:
            failures.append((q["id"], path, got, q["expected_url"]))

    if failures:
        print(f"\nERROR: mapper disagrees with {len(failures)} researched URLs:")
        for qid, path, got, want in failures:
            print(f"  {qid}  {path}\n      got  {got}\n      want {want}")
        print("\n  Check: has the TOC structure changed upstream, or is parse_toc wrong?")
        return 1
    print(f"urls        : {len(questions)}/{len(questions)} reproduced by the mapper")

    # --- 4. emit ------------------------------------------------------------
    out = {
        "questions": questions,
        "out_of_scope": data["out_of_scope"],
        "_toc_fixtures": fixtures,
        "_meta": {
            "source": "golden-draft.json, validated against AdobeDocs/experience-platform.en",
            "license": "Adobe documentation is MIT licensed; see README attribution",
            "corrections": corrected,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    print(f"out_of_scope: {len(data['out_of_scope'])} refusal probes")
    print(f"toc fixtures: {len(fixtures)} guides embedded ({', '.join(sorted(fixtures))})")
    print(f"corrections : {len(corrected)}")
    print(f"wrote       : {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
