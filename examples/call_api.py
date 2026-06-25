from pathlib import Path
import json

import requests


API_URL = "http://zystarsolver.meteoroid.fit/solve"
HERE = Path(__file__).resolve().parent


def call_sample(model: str, width: int, height: int, catalog: str):
    points = json.loads((HERE / f"sample_points_{model.replace('-', '_')}.json").read_text(encoding="utf-8"))
    payload = {
        "model": model,
        "width": width,
        "height": height,
        "catalog": catalog,
        "points": points,
    }
    response = requests.post(API_URL, json=payload, timeout=60)
    print(f"{model} sample")
    print("HTTP", response.status_code)
    print(response.text)


if __name__ == "__main__":
    call_sample("30-100", 1920, 1200, "HIP")
    print()
    call_sample("10-35", 2048, 2048, "GAIA")
