"""Policy-only subprocess worker for secure WBM execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("frozen", "random"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    state = json.loads(sys.stdin.read())
    queries = state["queries"]
    if not queries:
        raise ValueError("policy state contains no queries")
    if args.policy == "frozen":
        selected = min(
            queries,
            key=lambda item: (
                item["base_hull_distance_ev_per_atom"] / item["oracle_cost"],
                item["query_id"],
            ),
        )
    else:
        selected = min(
            queries,
            key=lambda item: (
                hashlib.sha256(f"{args.seed}:{item['query_id']}".encode()).hexdigest(),
                item["query_id"],
            ),
        )
    sys.stdout.write(selected["query_id"])


if __name__ == "__main__":
    main()
