"""Print an exact binary active-witness diagnostic; this is not materials evidence."""

from __future__ import annotations

import argparse
import json

from evimem.matmem.exact_dp import exact_policy_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=int, nargs="+", default=[3, 3, 2])
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=2)
    args = parser.parse_args()
    if args.budget < 1 or args.capacity < 0 or min(args.clusters) < 0:
        parser.error("budget must be positive; capacities and cluster counts non-negative")
    result = {
        "scope": "exact_binary_witness_diagnostic_not_materials_evidence",
        "remaining_by_cluster": args.clusters,
        "oracle_budget": args.budget,
        "active_witness_budget": args.capacity,
        "expected_discoveries": exact_policy_comparison(
            tuple(args.clusters),
            oracle_budget=args.budget,
            active_witness_budget=args.capacity,
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
