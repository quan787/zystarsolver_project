# ZYstarsolver

[中文](#中文)

ZYstarsolver identifies catalog stars from brightness-sorted 2D image points.
Use the hosted API for quick trials, or run the ONNX model locally for regular
processing.

## Quick Start: Hosted API

Endpoint:

```text
POST http://zystarsolver.meteoroid.fit/solve
```

Minimal Python call:

```python
from pathlib import Path
import json
import requests

points = json.loads(Path("examples/sample_points_30_100.json").read_text())
payload = {
    "model": "30-100",
    "width": 1920,
    "height": 1200,
    "catalog": "HIP",
    "points": points,
}

response = requests.post("http://zystarsolver.meteoroid.fit/solve", json=payload, timeout=60)
print(response.status_code)
print(response.json())
```

The hosted service is a temporary free trial with a limit of 100 requests per
source IP per day. Use local inference for routine or high-volume work.

## Install

Install from GitHub:

```bash
pip install git+ssh://git@github.com/quan787/zystarsolver_project.git
```

Or clone and install locally:

```bash
git clone git@github.com:quan787/zystarsolver_project.git
cd zystarsolver_project
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

Download model packages:

```bash
python scripts/download_models.py  # download both models from the default release
python scripts/download_models.py --model 30-100  # download one model
python scripts/download_models.py --model 10-35 --url https://example.com/10-35.zip
python scripts/download_models.py --model 30-100 --archive C:\path\to\30-100.zip
```

If repository scripts are not available after installation, use:

```bash
zystarsolver-download-models --model 30-100
```

Downloads are written to `models/` under the current working directory unless
`--model-dir` is provided.

## Local Inference

Recommended object style:

```python
from ZYstarsolver import StarSolver

solver = StarSolver(model="30-100")
try:
    result = solver.predict(
        width=1920,
        height=1200,
        points=[[1231.9, 438.9], [1231.6, 438.7]],
        catalog="HIP",
    )
finally:
    solver.close()

print(result["results"])
```

Context manager style:

```python
from ZYstarsolver import StarSolver

with StarSolver(model="30-100") as solver:
    result = solver.predict(width=1920, height=1200, points=points, catalog="HIP")
```

Batch local inference uses one model, width, height, and catalog for all samples;
`points_batch` contains multiple `[[x, y], ...]` point lists. Large batches are
processed in chunks; the default `batch_size` is 256.

```python
with StarSolver(model="30-100") as solver:
    result = solver.predict_batch(
        width=1920,
        height=1200,
        points_batch=[points1, points2],
        catalog="HIP",
    )
```

Use a custom model directory:

```python
solver = StarSolver(model="10-35", model_dir=r"C:\models")
```

## Parameters

| Parameter | Required | Description |
| --- | --- | --- |
| `model` | yes | `"10-35"` for about 10-35 degree horizontal FOV, or `"30-100"` for about 30-100 degree horizontal FOV. |
| `width` | yes | Image width in pixels. |
| `height` | yes | Image height in pixels. |
| `points` | yes | Raw pixel coordinates as `[[x, y], ...]`, sorted from brightest to faintest. |
| `points_batch` | local batch only | Multiple `points` lists for local `predict_batch()`. |
| `batch_size` | local batch only | Optional local chunk size for `predict_batch()`. Defaults to 256. |
| `catalog` | no | `"HIP"` or `"GAIA"`. Defaults to `"HIP"`. This is a catalog preference, not a guarantee that every returned ID uses that catalog. |
| `model_dir` | local only | Optional local model directory for `StarSolver`. |

The local response contains one result per input point:

```json
{
  "results": [
    {"catalog_id": "HIP24608", "probability": 0.9999960660934448}
  ]
}
```

The hosted API returns the same `results` field and also includes `ok`, `model`,
and `catalog`. Results are returned in the same order as input points. Empty
`catalog_id` means no reliable catalog match for that input point. A low probability means the match is uncertain.

Some stars only have one available catalog identifier. Even when `catalog` is set to `HIP` or `GAIA`, the response can still contain IDs from the other catalog for those stars.

## Input Data Quality

- Choose `10-35` for horizontal FOV about 10-35 degrees; provide 25-40 stars.
- Choose `30-100` for horizontal FOV about 30-100 degrees; provide 20-30 stars.
- Use near-full input counts when possible: 40 stars for `10-35`, 30 stars for `30-100`.
- Points must be raw pixel coordinates `[x, y]`, measured in the same image coordinate system as `width` and `height`.
- Sort points by image brightness from brightest to faintest. The model expects ordered star lists.
- Coordinates must satisfy `0 <= x <= width` and `0 <= y <= height`.
- Dropped stars and false stars should ideally each be within about 15%.
- Photometric sorting error should preferably be within about 0.2 magnitude.
- Lower centroid noise improves reliability.
- Landscape images usually work better. For portrait images, rotating before extracting points is recommended.
- Keep aspect ratio within about 2:1 when possible.

## Licenses

- Source code: MIT License.
- ONNX model weights: CC BY-NC 4.0.

See `LICENSE` and `MODEL_LICENSE.md`.

## Recommended Citation

## 中文

ZYstarsolver 根据按亮度排序的二维图像星点识别星表恒星。你可以用托管 API
快速试用，也可以下载 ONNX 模型在本地运行。

## 快速上手：托管 API

接口地址：

```text
POST http://zystarsolver.meteoroid.fit/solve
```

最小 Python 调用示例：

```python
from pathlib import Path
import json
import requests

points = json.loads(Path("examples/sample_points_30_100.json").read_text())
payload = {
    "model": "30-100",
    "width": 1920,
    "height": 1200,
    "catalog": "HIP",
    "points": points,
}

response = requests.post("http://zystarsolver.meteoroid.fit/solve", json=payload, timeout=60)
print(response.status_code)
print(response.json())
```

托管服务是临时免费试用接口，当前限制为每个来源 IP 每天 100 次请求。长期或
高频使用建议在本地推理。

## 安装

从 GitHub 安装：

```bash
pip install git+ssh://git@github.com/quan787/zystarsolver_project.git
```

或者 clone 后本地安装：

```bash
git clone git@github.com:quan787/zystarsolver_project.git
cd zystarsolver_project
pip install .
```

开发安装：

```bash
pip install -e ".[dev]"
```

下载模型：

```bash
python scripts/download_models.py  # 从默认 release 下载两个模型
python scripts/download_models.py --model 30-100  # 只下载一个模型
python scripts/download_models.py --model 10-35 --url https://example.com/10-35.zip
python scripts/download_models.py --model 30-100 --archive C:\path\to\30-100.zip
```

如果安装后没有仓库脚本，可以使用：

```bash
zystarsolver-download-models --model 30-100
```

默认下载到当前工作目录下的 `models/`，也可以用 `--model-dir` 指定目录。

## 本地推理

推荐写法：

```python
from ZYstarsolver import StarSolver

solver = StarSolver(model="30-100")
try:
    result = solver.predict(
        width=1920,
        height=1200,
        points=[[1231.9, 438.9], [1231.6, 438.7]],
        catalog="HIP",
    )
finally:
    solver.close()

print(result["results"])
```

也可以使用 `with`：

```python
from ZYstarsolver import StarSolver

with StarSolver(model="30-100") as solver:
    result = solver.predict(width=1920, height=1200, points=points, catalog="HIP")
```

本地批量推理对所有样本使用同一个模型、图像宽高和星表偏好；`points_batch`
包含多个 `[[x, y], ...]` 星点列表。大批量输入会按块处理，默认 `batch_size`
为 256。

```python
with StarSolver(model="30-100") as solver:
    result = solver.predict_batch(
        width=1920,
        height=1200,
        points_batch=[points1, points2],
        catalog="HIP",
    )
```

指定模型目录：

```python
solver = StarSolver(model="10-35", model_dir=r"C:\models")
```

## 参数

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `model` | 是 | `"10-35"` 适用于约 10-35 度水平视场，`"30-100"` 适用于约 30-100 度水平视场。 |
| `width` | 是 | 图像宽度，单位像素。 |
| `height` | 是 | 图像高度，单位像素。 |
| `points` | 是 | 原始像素坐标 `[[x, y], ...]`，按亮度从亮到暗排序。 |
| `points_batch` | 仅本地批量 | 多个 `points` 列表，用于本地 `predict_batch()`。 |
| `batch_size` | 仅本地批量 | `predict_batch()` 的分块大小，默认 256。 |
| `catalog` | 否 | `"HIP"` 或 `"GAIA"`，默认 `"HIP"`。这是星表偏好，不保证每个返回编号都来自该星表。 |
| `model_dir` | 仅本地 | `StarSolver` 使用的本地模型目录。 |

本地推理返回每个输入点对应的结果：

```json
{
  "results": [
    {"catalog_id": "HIP24608", "probability": 0.9999960660934448}
  ]
}
```

托管 API 也返回同样的 `results` 字段，并额外包含 `ok`、`model` 和 `catalog`。
结果顺序与输入点顺序一致。空的 `catalog_id` 表示该输入点没有可靠星表匹配。
较低的 probability 表示匹配不确定。

有些恒星只有一种星表编号。即使指定了 `catalog` 为 `HIP` 或 `GAIA`，这些
恒星仍然可能返回另一种星表的编号。

## 输入数据质量

- `10-35` 适用于约 10-35 度水平视场，输入 25-40 颗星。
- `30-100` 适用于约 30-100 度水平视场，输入 20-30 颗星。
- 尽量使用接近上限的输入星点数：`10-35` 使用 40 颗，`30-100` 使用 30 颗。
- 点坐标必须是原始像素坐标 `[x, y]`，并与 `width`、`height` 使用同一个图像坐标系。
- 按图像亮度从亮到暗排序。模型期望输入星点列表已经排序。
- 坐标必须满足 `0 <= x <= width` 和 `0 <= y <= height`。
- 漏检星和误检星最好分别控制在约 15% 以内。
- 测光排序误差最好控制在约 0.2 等以内。
- 质心误差越低，结果通常越稳定。
- 横向图像通常效果更好；竖图建议先旋转后再提取星点。
- 宽高比尽量控制在约 2:1 以内。

## 许可

- 源代码：MIT License。
- ONNX 模型权重：CC BY-NC 4.0。

见 `LICENSE` 和 `MODEL_LICENSE.md`。

## 推荐引用
