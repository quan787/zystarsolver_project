import argparse
from pathlib import Path
import json
import sys


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ZYstarsolver import StarSolver


def default_dimensions(model: str) -> tuple[int, int, str]:
    if model == "10-35":
        return 2048, 2048, "GAIA"
    return 1920, 1200, "HIP"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one local ZYstarsolver sample.")
    parser.add_argument("--model", choices=["10-35", "30-100"], default="30-100")
    parser.add_argument("--model-dir", help="Optional model directory or parent models directory.")
    parser.add_argument("--points", help="Optional JSON file containing [[x, y], ...] points.")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--catalog", choices=["HIP", "GAIA"])
    args = parser.parse_args()

    width, height, catalog = default_dimensions(args.model)
    width = args.width or width
    height = args.height or height
    catalog = args.catalog or catalog
    points_path = Path(args.points) if args.points else HERE / f"sample_points_{args.model.replace('-', '_')}.json"
    points = json.loads(points_path.read_text(encoding="utf-8"))

    with StarSolver(model=args.model, model_dir=args.model_dir) as solver:
        response = solver.predict(width=width, height=height, points=points, catalog=catalog)
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
