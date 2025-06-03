import argparse
import asyncio

from eval.strategic_reasoning import run_strategic_reasoning
from eval.decision_making import run_decision_making


def parse_args():
    parser = argparse.ArgumentParser(description="VS-Bench experiment launcher")
    parser.add_argument("--eval", required=True, default="decision-making",
                        choices=["strategic_reasoning", "decision-making"], help="evaluation")
    parser.add_argument("--exp", required=True, help="experiment name")
    return parser.parse_args()


async def main():
    args = parse_args()
    if args.eval == "strategic_reasoning":
        await run_strategic_reasoning(args.exp)
    elif args.eval == "decision-making":
        await run_decision_making(args.exp)


if __name__ == "__main__":
    asyncio.run(main())
