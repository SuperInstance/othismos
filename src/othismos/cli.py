#!/usr/bin/env python3
"""
othismos — CLI for measuring constraint pressure in bounded systems.

Usage:
    othismos pressure <model_dir> [--constraints config.yaml]
    othismos reef <action> [--db reef.json]
    othismos reef add <id> <content> [--refs id1,id2]
    othismos reef list [--layer surface|consolidation|foundation]
    othismos reef fail <id>
    othismos reef stats
    othismos diagnose <history.json> [--heat 1.0]
    othismos version
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from othismos import (
    PressureGauge,
    PopcornDiagnostic,
    Reef,
    ReefLayer,
    Reefquake,
    SystemHealth,
    __version__,
    pressure_summary,
    save_history,
    load_history,
)


# ─── Pressure Commands ────────────────────────────────────────────

def cmd_pressure(args):
    """Analyze pressure data from a saved history."""
    if not args.history:
        print("Error: --history required (JSON file from save_history)")
        sys.exit(1)

    gauge = load_history(args.history)
    summary = pressure_summary(gauge)

    print("\n=== Pressure Analysis ===\n")
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sk, sv in v.items():
                print(f"    {sk}: {sv:.6f}")
        else:
            print(f"  {k}: {v}")


# ─── Reef Commands ────────────────────────────────────────────────

REEF_DB_DEFAULT = ".othismos-reef.json"


def _load_reef(path: str) -> Reef:
    """Load a reef from a JSON snapshot (or create new)."""
    reef = Reef()
    p = Path(path)
    if p.exists():
        data = json.loads(p.read_text())
        for dep_data in data.get("deposits", []):
            reef.submit(
                dep_data["id"],
                dep_data["content"],
                references=dep_data.get("references", []),
            )
            # Restore age and layer
            if dep_data.get("age"):
                dep = reef.query(dep_data["id"])
                if dep:
                    dep.age = dep_data["age"]
    return reef


def _save_reef(reef: Reef, path: str):
    """Save a reef to JSON."""
    deposits = []
    for dep_id in list(reef._deposits.keys()):
        dep = reef._deposits[dep_id]
        deposits.append({
            "id": dep.id,
            "content": dep.content,
            "references": dep.references,
            "referenced_by": list(dep.referenced_by),
            "layer": dep.layer.name,
            "age": dep.age,
            "depth_score": dep.depth_score,
        })
    data = {
        "version": __version__,
        "step": reef.step,
        "deposits": deposits,
    }
    Path(path).write_text(json.dumps(data, indent=2))


def cmd_reef(args):
    """Manage a local knowledge reef."""
    db_path = args.db or REEF_DB_DEFAULT

    if args.action == "add":
        reef = _load_reef(db_path)
        refs = args.refs.split(",") if args.refs else []
        accepted, msg = reef.submit(args.id, args.content, references=refs)
        if accepted:
            _save_reef(reef, db_path)
            print(f"✓ {msg}")
        else:
            print(f"✗ {msg}")
            sys.exit(1)

    elif args.action == "list":
        reef = _load_reef(db_path)
        layer_filter = args.layer.upper() if args.layer else None

        deposits = []
        for dep in reef._deposits.values():
            if layer_filter and dep.layer.name != layer_filter:
                continue
            deposits.append(dep)

        deposits.sort(key=lambda d: (-d.depth_score, d.id))

        if not deposits:
            print("(empty reef)")
            return

        print(f"\n{'ID':<20} {'Layer':<15} {'Refs':<6} {'Age':<6} {'Depth':<8} Content")
        print("-" * 90)
        for dep in deposits:
            content_preview = dep.content[:40] + ("..." if len(dep.content) > 40 else "")
            print(f"{dep.id:<20} {dep.layer.name:<15} {dep.reference_count:<6} {dep.age:<6} {dep.depth_score:<8.1f} {content_preview}")

    elif args.action == "fail":
        reef = _load_reef(db_path)
        try:
            reef.fail_deposit(args.id)
        except Reefquake as rq:
            _save_reef(reef, db_path)
            print(f"⚠️  REEFQUAKE: deposit '{rq.failed_id}' failed!")
            print(f"   {len(rq.affected)} deposits removed:")
            for dep_id in rq.affected:
                print(f"   - {dep_id}")
        except KeyError:
            print(f"✗ Unknown deposit: {args.id}")
            sys.exit(1)

    elif args.action == "stats":
        reef = _load_reef(db_path)
        summary = reef.summary()
        print("\n=== Reef Statistics ===\n")
        for k, v in summary.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for sk, sv in v.items():
                    print(f"    {sk}: {sv}")
            else:
                print(f"  {k}: {v}")

    elif args.action == "tick":
        reef = _load_reef(db_path)
        result = reef.tick()
        _save_reef(reef, db_path)
        print(f"Step {result['step']}: {result['total_deposits']} deposits")
        if result["eroded"]:
            print(f"  Eroded: {', '.join(result['eroded'])}")
        if result["promoted"]:
            for dep_id, layer in result["promoted"]:
                print(f"  Promoted: {dep_id} → {layer}")

    elif args.action == "search":
        reef = _load_reef(db_path)
        results = reef.search(args.query)
        if not results:
            print("(no results)")
            return
        for dep in results:
            print(f"\n  {dep.id} (depth={dep.depth_score:.1f}, layer={dep.layer.name})")
            print(f"  {dep.content[:200]}")

    elif args.action == "graph":
        reef = _load_reef(db_path)
        graph = reef.citation_graph()
        print(json.dumps(graph, indent=2))

    else:
        print(f"Unknown reef action: {args.action}")
        sys.exit(1)


# ─── Diagnose Commands ────────────────────────────────────────────

def cmd_diagnose(args):
    """Run popcorn diagnostic on saved pressure data."""
    gauge = load_history(args.history)
    pressures = [m.pressure for m in gauge.history]

    diag = PopcornDiagnostic()
    result = diag.diagnose(pressures, heat=args.heat)

    health_emoji = {
        SystemHealth.POP: "🍿",
        SystemHealth.BURN: "🔥",
        SystemHealth.SEEP: "💧",
        SystemHealth.DORMANT: "😴",
    }

    print(f"\n{health_emoji.get(result.health, '?')} Health: {result.health.value.upper()}")
    print(f"   Confidence: {result.confidence:.0%}")
    print(f"   Pressure: {result.pressure:.6f}")
    print(f"   Heat: {result.heat:.6f}")
    print(f"   Efficiency: {result.pressure_efficiency:.4f}")
    print()
    for signal in result.signals:
        print(f"   → {signal}")
    print()
    print(f"   {result.recommendation}")


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="othismos",
        description="Measure constraint pressure in bounded systems. The push IS the knowing.",
    )
    sub = parser.add_subparsers(dest="command")

    # Pressure
    p_pressure = sub.add_parser("pressure", help="Analyze pressure history")
    p_pressure.add_argument("--history", help="JSON file from save_history")
    p_pressure.set_defaults(func=cmd_pressure)

    # Reef
    p_reef = sub.add_parser("reef", help="Manage a knowledge reef")
    p_reef.add_argument("action", choices=["add", "list", "fail", "stats", "tick", "search", "graph"])
    p_reef.add_argument("id", nargs="?", help="Deposit ID (for add/fail)")
    p_reef.add_argument("content", nargs="?", help="Deposit content (for add)")
    p_reef.add_argument("--refs", help="Comma-separated reference IDs (for add)")
    p_reef.add_argument("--layer", choices=["surface", "consolidation", "foundation"])
    p_reef.add_argument("--query", help="Search query (for search action)")
    p_reef.add_argument("--db", help=f"Reef database path (default: {REEF_DB_DEFAULT})")
    p_reef.set_defaults(func=cmd_reef)

    # Diagnose
    p_diag = sub.add_parser("diagnose", help="Run popcorn diagnostic")
    p_diag.add_argument("history", help="JSON file from save_history")
    p_diag.add_argument("--heat", type=float, default=1.0, help="Current external pressure")
    p_diag.set_defaults(func=cmd_diagnose)

    # Version
    p_version = sub.add_parser("version", help="Print version")
    p_version.set_defaults(func=lambda a: print(f"othismos {__version__}"))

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
