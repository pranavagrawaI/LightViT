"""Generate deck-ready SVG charts from recovered checkpoint metrics."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import html
import math

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"


COLORS = {
    "pure_svd": "#1f77b4",
    "act_svd": "#2ca02c",
    "hybrid_tucker": "#d62728",
    "hybrid_tensorly": "#9467bd",
    "baseline": "#111111",
}
LABELS = {
    "pure_svd": "Pure SVD",
    "act_svd": "Act-SVD",
    "hybrid_tucker": "Hybrid Tucker",
    "hybrid_tensorly": "Hybrid Tensorly",
    "baseline": "Baseline",
}
ORDER = {
    "pure_svd": 0,
    "act_svd": 1,
    "hybrid_tucker": 2,
    "hybrid_tensorly": 3,
}
HIGHLIGHTS = {
    ("pure_svd", 0.5): "best acc",
    ("hybrid_tucker", 0.5): "best tradeoff",
    ("pure_svd", 0.25): "compact",
    ("hybrid_tucker", 0.25): "knee",
}


@dataclass(frozen=True)
class ModelPoint:
    family: str
    rank: float
    params: int
    model_size_mb: float
    checkpoint_mb: float
    latency_ms: float
    pre_recovery_acc: float
    best_acc: float
    final_acc: float
    metrics_path: Path


@dataclass(frozen=True)
class BaselinePoint:
    model_size_mb: float
    checkpoint_mb: float
    latency_ms: float
    accuracy: float


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def as_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def find_baseline() -> BaselinePoint:
    manifest = CHECKPOINTS_DIR / "pure_svd" / "manifest.csv"
    for row in read_rows(manifest):
        if row.get("stage") == "baseline":
            return BaselinePoint(
                model_size_mb=as_float(row["model_size_mb"]),
                checkpoint_mb=as_float(row["checkpoint_mb"]),
                latency_ms=as_float(row["latency_ms"]),
                accuracy=as_float(row["top1_accuracy"]),
            )
    raise ValueError(f"No baseline row found in {manifest}")


def recovery_points() -> list[ModelPoint]:
    points: list[ModelPoint] = []
    for metrics_path in sorted(CHECKPOINTS_DIR.glob("*_recovered/*_recovery_metrics.csv")):
        rows = read_rows(metrics_path)
        pre = next((row for row in rows if row.get("stage") == "pre_recovery"), None)
        recovery = [row for row in rows if row.get("stage") == "recovery_epoch"]
        if pre is None or not recovery:
            continue
        best = max(recovery, key=lambda row: as_float(row["top1_accuracy"]))
        final = recovery[-1]
        family = best["family"]
        if metrics_path.parent.name.startswith("hybrid_tensorly"):
            family = "hybrid_tensorly"
        points.append(
            ModelPoint(
                family=family,
                rank=as_float(best["rank_ratio"]),
                params=as_int(best["params"]),
                model_size_mb=as_float(best["model_size_mb"]),
                checkpoint_mb=as_float(best["checkpoint_mb"]),
                latency_ms=as_float(best["latency_ms"]),
                pre_recovery_acc=as_float(pre["top1_accuracy"]),
                best_acc=as_float(best["top1_accuracy"]),
                final_acc=as_float(final["top1_accuracy"]),
                metrics_path=metrics_path,
            )
        )
    return sorted(points, key=lambda point: (ORDER.get(point.family, 99), point.rank))


def write_summary_csv(points: list[ModelPoint], baseline: BaselinePoint) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "chart_data_summary.csv"
    fieldnames = [
        "family",
        "rank",
        "params",
        "model_size_mb",
        "checkpoint_mb",
        "latency_ms",
        "pre_recovery_acc",
        "best_acc",
        "final_acc",
        "metrics_path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "family": "baseline",
                "rank": 1.0,
                "params": "",
                "model_size_mb": f"{baseline.model_size_mb:.4f}",
                "checkpoint_mb": f"{baseline.checkpoint_mb:.4f}",
                "latency_ms": f"{baseline.latency_ms:.4f}",
                "pre_recovery_acc": "",
                "best_acc": f"{baseline.accuracy:.4f}",
                "final_acc": f"{baseline.accuracy:.4f}",
                "metrics_path": CHECKPOINTS_DIR / "pure_svd" / "manifest.csv",
            }
        )
        for point in points:
            writer.writerow(
                {
                    "family": point.family,
                    "rank": f"{point.rank:.4f}",
                    "params": point.params,
                    "model_size_mb": f"{point.model_size_mb:.4f}",
                    "checkpoint_mb": f"{point.checkpoint_mb:.4f}",
                    "latency_ms": f"{point.latency_ms:.4f}",
                    "pre_recovery_acc": f"{point.pre_recovery_acc:.4f}",
                    "best_acc": f"{point.best_acc:.4f}",
                    "final_acc": f"{point.final_acc:.4f}",
                    "metrics_path": point.metrics_path,
                }
            )


def nice_ticks(min_value: float, max_value: float, count: int = 6) -> list[float]:
    if min_value == max_value:
        return [min_value]
    raw_step = (max_value - min_value) / max(1, count - 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    if normalized <= 1:
        step = magnitude
    elif normalized <= 2:
        step = 2 * magnitude
    elif normalized <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    start = math.floor(min_value / step) * step
    end = math.ceil(max_value / step) * step
    ticks = []
    value = start
    while value <= end + step * 0.5:
        ticks.append(round(value, 10))
        value += step
    return ticks


class SvgChart:
    def __init__(
        self,
        title: str,
        x_label: str,
        y_label: str,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        width: int = 1200,
        height: int = 760,
    ) -> None:
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.width = width
        self.height = height
        self.left = 92
        self.right = 280
        self.top = 78
        self.bottom = 92
        self.items: list[str] = []

    @property
    def plot_width(self) -> int:
        return self.width - self.left - self.right

    @property
    def plot_height(self) -> int:
        return self.height - self.top - self.bottom

    def x(self, value: float) -> float:
        return self.left + (value - self.x_min) / (self.x_max - self.x_min) * self.plot_width

    def y(self, value: float) -> float:
        return self.top + (self.y_max - value) / (self.y_max - self.y_min) * self.plot_height

    def text(self, x: float, y: float, text: str, size: int = 16, anchor: str = "start", weight: str = "400") -> None:
        safe = html.escape(text)
        self.items.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
            f'font-family="Inter, Arial, sans-serif" font-weight="{weight}" fill="#1f2933">{safe}</text>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str, width: float = 2.0, dash: str | None = None) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}"{dash_attr} />'
        )

    def polyline(self, points: list[tuple[float, float]], color: str, width: float = 3.0, dash: str | None = None) -> None:
        if len(points) < 2:
            return
        encoded = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(
            f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-linejoin="round" stroke-linecap="round"{dash_attr} />'
        )

    def circle(self, x: float, y: float, radius: float, color: str, stroke: str = "white", stroke_width: float = 2.0) -> None:
        self.items.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}" />'
        )

    def diamond(self, x: float, y: float, size: float, color: str) -> None:
        points = [
            (x, y - size),
            (x + size, y),
            (x, y + size),
            (x - size, y),
        ]
        encoded = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
        self.items.append(
            f'<polygon points="{encoded}" fill="{color}" stroke="white" stroke-width="2.5" />'
        )

    def axes(self, x_ticks: list[float] | None = None, y_ticks: list[float] | None = None) -> None:
        x_ticks = nice_ticks(self.x_min, self.x_max) if x_ticks is None else x_ticks
        y_ticks = nice_ticks(self.y_min, self.y_max) if y_ticks is None else y_ticks
        self.items.append(
            f'<rect x="0" y="0" width="{self.width}" height="{self.height}" fill="#ffffff" />'
        )
        self.text(self.width / 2, 36, self.title, size=24, anchor="middle", weight="700")
        self.line(self.left, self.top + self.plot_height, self.left + self.plot_width, self.top + self.plot_height, "#334155", 1.5)
        self.line(self.left, self.top, self.left, self.top + self.plot_height, "#334155", 1.5)
        for tick in x_ticks:
            if tick < self.x_min or tick > self.x_max:
                continue
            x = self.x(tick)
            self.line(x, self.top, x, self.top + self.plot_height, "#e2e8f0", 1.0)
            self.text(x, self.top + self.plot_height + 28, format_tick(tick), size=13, anchor="middle")
        for tick in y_ticks:
            if tick < self.y_min or tick > self.y_max:
                continue
            y = self.y(tick)
            self.line(self.left, y, self.left + self.plot_width, y, "#e2e8f0", 1.0)
            self.text(self.left - 12, y + 5, format_tick(tick), size=13, anchor="end")
        self.text(self.left + self.plot_width / 2, self.height - 28, self.x_label, size=16, anchor="middle", weight="600")
        self.items.append(
            f'<text x="26" y="{self.top + self.plot_height / 2:.1f}" font-size="16" text-anchor="middle" '
            f'font-family="Inter, Arial, sans-serif" font-weight="600" fill="#1f2933" '
            f'transform="rotate(-90 26 {self.top + self.plot_height / 2:.1f})">{html.escape(self.y_label)}</text>'
        )

    def legend(self, families: list[str], baseline: bool = True) -> None:
        x = self.left + self.plot_width + 38
        y = self.top + 28
        self.text(x, y - 12, "Method", size=15, weight="700")
        for family in families:
            self.circle(x + 9, y + 9, 7, COLORS[family], stroke="#ffffff")
            self.text(x + 28, y + 14, LABELS[family], size=14)
            y += 30
        if baseline:
            self.diamond(x + 9, y + 9, 8, COLORS["baseline"])
            self.text(x + 28, y + 14, LABELS["baseline"], size=14)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(self.items)
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">\n{content}\n</svg>\n'
        )


def hex_rgb(color: str) -> tuple[int, int, int]:
    if color == "none":
        return (255, 255, 255)
    if color == "white":
        return (255, 255, 255)
    if color == "black":
        return (0, 0, 0)
    color = color.lstrip("#")
    return tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))


def font(size: int, weight: str = "400") -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    path = names[0] if weight == "700" else names[1]
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


class PngChart(SvgChart):
    def __init__(self, *args, scale: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scale = scale
        self.image = Image.new("RGB", (self.width * scale, self.height * scale), "white")
        self.draw = ImageDraw.Draw(self.image)

    def sx(self, value: float) -> int:
        return round(value * self.scale)

    def sxy(self, x: float, y: float) -> tuple[int, int]:
        return self.sx(x), self.sx(y)

    def text(self, x: float, y: float, text: str, size: int = 16, anchor: str = "start", weight: str = "400") -> None:
        draw_font = font(size * self.scale, weight)
        px, py = self.sxy(x, y)
        bbox = self.draw.textbbox((0, 0), text, font=draw_font)
        width = bbox[2] - bbox[0]
        if anchor == "middle":
            px -= width // 2
        elif anchor == "end":
            px -= width
        self.draw.text((px, py - round(size * self.scale * 0.82)), text, fill=hex_rgb("#1f2933"), font=draw_font)

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str, width: float = 2.0, dash: str | None = None) -> None:
        if dash:
            self._dashed_line((x1, y1), (x2, y2), color, width, dash)
            return
        self.draw.line(
            [self.sxy(x1, y1), self.sxy(x2, y2)],
            fill=hex_rgb(color),
            width=max(1, self.sx(width)),
        )

    def _dashed_line(self, start: tuple[float, float], end: tuple[float, float], color: str, width: float, dash: str) -> None:
        pattern = [float(part) for part in dash.split()]
        if not pattern:
            pattern = [6.0, 6.0]
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return
        ux = dx / length
        uy = dy / length
        distance = 0.0
        index = 0
        while distance < length:
            segment = pattern[index % len(pattern)]
            next_distance = min(length, distance + segment)
            if index % 2 == 0:
                self.draw.line(
                    [
                        self.sxy(x1 + ux * distance, y1 + uy * distance),
                        self.sxy(x1 + ux * next_distance, y1 + uy * next_distance),
                    ],
                    fill=hex_rgb(color),
                    width=max(1, self.sx(width)),
                )
            distance = next_distance
            index += 1

    def polyline(self, points: list[tuple[float, float]], color: str, width: float = 3.0, dash: str | None = None) -> None:
        if len(points) < 2:
            return
        if dash:
            for start, end in zip(points, points[1:]):
                self._dashed_line(start, end, color, width, dash)
            return
        self.draw.line(
            [self.sxy(x, y) for x, y in points],
            fill=hex_rgb(color),
            width=max(1, self.sx(width)),
            joint="curve",
        )

    def circle(self, x: float, y: float, radius: float, color: str, stroke: str = "white", stroke_width: float = 2.0) -> None:
        box = [
            self.sx(x - radius),
            self.sx(y - radius),
            self.sx(x + radius),
            self.sx(y + radius),
        ]
        fill = None if color == "none" else hex_rgb(color)
        self.draw.ellipse(box, fill=fill, outline=hex_rgb(stroke), width=max(1, self.sx(stroke_width)))

    def diamond(self, x: float, y: float, size: float, color: str) -> None:
        points = [
            self.sxy(x, y - size),
            self.sxy(x + size, y),
            self.sxy(x, y + size),
            self.sxy(x - size, y),
        ]
        self.draw.polygon(points, fill=hex_rgb(color), outline=hex_rgb("ffffff"))

    def axes(self, x_ticks: list[float] | None = None, y_ticks: list[float] | None = None) -> None:
        x_ticks = nice_ticks(self.x_min, self.x_max) if x_ticks is None else x_ticks
        y_ticks = nice_ticks(self.y_min, self.y_max) if y_ticks is None else y_ticks
        self.text(self.width / 2, 36, self.title, size=24, anchor="middle", weight="700")
        self.line(self.left, self.top + self.plot_height, self.left + self.plot_width, self.top + self.plot_height, "#334155", 1.5)
        self.line(self.left, self.top, self.left, self.top + self.plot_height, "#334155", 1.5)
        for tick in x_ticks:
            if tick < self.x_min or tick > self.x_max:
                continue
            x = self.x(tick)
            self.line(x, self.top, x, self.top + self.plot_height, "#e2e8f0", 1.0)
            self.text(x, self.top + self.plot_height + 28, format_tick(tick), size=13, anchor="middle")
        for tick in y_ticks:
            if tick < self.y_min or tick > self.y_max:
                continue
            y = self.y(tick)
            self.line(self.left, y, self.left + self.plot_width, y, "#e2e8f0", 1.0)
            self.text(self.left - 12, y + 5, format_tick(tick), size=13, anchor="end")
        self.text(self.left + self.plot_width / 2, self.height - 28, self.x_label, size=16, anchor="middle", weight="600")
        self._rotated_y_label()

    def _rotated_y_label(self) -> None:
        draw_font = font(16 * self.scale, "700")
        bbox = self.draw.textbbox((0, 0), self.y_label, font=draw_font)
        text_image = Image.new("RGBA", (bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8), (255, 255, 255, 0))
        text_draw = ImageDraw.Draw(text_image)
        text_draw.text((4, 4), self.y_label, fill=hex_rgb("#1f2933"), font=draw_font)
        rotated = text_image.rotate(90, expand=True)
        x = self.sx(26) - rotated.width // 2
        y = self.sx(self.top + self.plot_height / 2) - rotated.height // 2
        self.image.paste(rotated.convert("RGB"), (x, y), rotated)

    def legend(self, families: list[str], baseline: bool = True) -> None:
        x = self.left + self.plot_width + 38
        y = self.top + 28
        self.text(x, y - 12, "Method", size=15, weight="700")
        for family in families:
            self.circle(x + 9, y + 9, 7, COLORS[family], stroke="#ffffff")
            self.text(x + 28, y + 14, LABELS[family], size=14)
            y += 30
        if baseline:
            self.diamond(x + 9, y + 9, 8, COLORS["baseline"])
            self.text(x + 28, y + 14, LABELS["baseline"], size=14)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        output = self.image.resize((self.width, self.height), Image.Resampling.LANCZOS)
        output.save(path.with_suffix(".png"))


class Chart:
    def __init__(self, *args, **kwargs) -> None:
        self.svg = SvgChart(*args, **kwargs)
        self.png = PngChart(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self.svg, name)

    def text(self, *args, **kwargs) -> None:
        self.svg.text(*args, **kwargs)
        self.png.text(*args, **kwargs)

    def line(self, *args, **kwargs) -> None:
        self.svg.line(*args, **kwargs)
        self.png.line(*args, **kwargs)

    def polyline(self, *args, **kwargs) -> None:
        self.svg.polyline(*args, **kwargs)
        self.png.polyline(*args, **kwargs)

    def circle(self, *args, **kwargs) -> None:
        self.svg.circle(*args, **kwargs)
        self.png.circle(*args, **kwargs)

    def diamond(self, *args, **kwargs) -> None:
        self.svg.diamond(*args, **kwargs)
        self.png.diamond(*args, **kwargs)

    def axes(self, *args, **kwargs) -> None:
        self.svg.axes(*args, **kwargs)
        self.png.axes(*args, **kwargs)

    def legend(self, *args, **kwargs) -> None:
        self.svg.legend(*args, **kwargs)
        self.png.legend(*args, **kwargs)

    def write(self, path: Path) -> None:
        self.svg.write(path)
        self.png.write(path)


def format_tick(value: float) -> str:
    if abs(value) >= 10 or value == 0:
        return f"{value:.0f}"
    return f"{value:.1f}"


def grouped(points: list[ModelPoint]) -> dict[str, list[ModelPoint]]:
    groups: dict[str, list[ModelPoint]] = {}
    for point in points:
        groups.setdefault(point.family, []).append(point)
    return {
        family: sorted(group, key=lambda point: point.rank)
        for family, group in sorted(groups.items(), key=lambda item: ORDER.get(item[0], 99))
    }


def label_point(chart: Chart, x: float, y: float, label: str, dx: float = 9, dy: float = -9) -> None:
    chart.text(chart.x(x) + dx, chart.y(y) + dy, label, size=12)


def chart_accuracy_vs_size(points: list[ModelPoint], baseline: BaselinePoint) -> None:
    x_max = max([baseline.model_size_mb, *[point.model_size_mb for point in points]]) * 1.08
    y_min = max(0, min(point.best_acc for point in points) - 4)
    y_max = baseline.accuracy + 2
    chart = Chart(
        "Accuracy vs Model Size",
        "Model size (MB, weights)",
        "Best top-1 accuracy (%)",
        0,
        x_max,
        y_min,
        y_max,
    )
    chart.axes()
    groups = grouped(points)
    for family, group in groups.items():
        color = COLORS[family]
        coords = [(chart.x(point.model_size_mb), chart.y(point.best_acc)) for point in group]
        chart.polyline(coords, color)
        for point in group:
            chart.circle(chart.x(point.model_size_mb), chart.y(point.best_acc), 6, color)
    chart.diamond(chart.x(baseline.model_size_mb), chart.y(baseline.accuracy), 9, COLORS["baseline"])
    label_point(chart, baseline.model_size_mb, baseline.accuracy, "baseline", dx=-62, dy=-12)
    chart.legend(list(groups))
    chart.write(OUTPUT_DIR / "04_accuracy_vs_model_size.svg")


def chart_accuracy_vs_latency(points: list[ModelPoint], baseline: BaselinePoint) -> None:
    x_max = max([baseline.latency_ms, *[point.latency_ms for point in points]]) * 1.12
    y_min = max(0, min(point.best_acc for point in points) - 4)
    y_max = baseline.accuracy + 2
    chart = Chart(
        "Accuracy vs Inference Latency",
        "Latency (ms, lower is faster)",
        "Best top-1 accuracy (%)",
        0,
        x_max,
        y_min,
        y_max,
    )
    chart.axes()
    groups = grouped(points)
    for family, group in groups.items():
        color = COLORS[family]
        for point in group:
            chart.circle(chart.x(point.latency_ms), chart.y(point.best_acc), 6, color)
    chart.diamond(chart.x(baseline.latency_ms), chart.y(baseline.accuracy), 9, COLORS["baseline"])
    label_point(chart, baseline.latency_ms, baseline.accuracy, "baseline", dx=10, dy=-12)
    chart.text(chart.left + 10, chart.top + 24, "Smaller models are not always faster.", size=14, weight="700")
    chart.legend(list(groups))
    chart.write(OUTPUT_DIR / "05_accuracy_vs_latency.svg")


def chart_rank_vs_accuracy(points: list[ModelPoint], baseline: BaselinePoint) -> None:
    y_min = max(0, min(point.best_acc for point in points) - 4)
    y_max = baseline.accuracy + 2
    chart = Chart(
        "Rank Fraction vs Best Accuracy",
        "Rank fraction",
        "Best top-1 accuracy (%)",
        0.1,
        0.52,
        y_min,
        y_max,
    )
    chart.axes(x_ticks=[0.125, 0.1875, 0.25, 0.375, 0.5])
    groups = grouped(points)
    chart.line(chart.left, chart.y(baseline.accuracy), chart.left + chart.plot_width, chart.y(baseline.accuracy), COLORS["baseline"], 2, "6 6")
    for family, group in groups.items():
        color = COLORS[family]
        coords = [(chart.x(point.rank), chart.y(point.best_acc)) for point in group]
        chart.polyline(coords, color)
        for point in group:
            chart.circle(chart.x(point.rank), chart.y(point.best_acc), 6, color)
    chart.text(chart.left + chart.plot_width - 4, chart.y(baseline.accuracy) - 8, "baseline", size=12, anchor="end")
    chart.legend(list(groups), baseline=False)
    chart.write(OUTPUT_DIR / "06_rank_vs_best_accuracy.svg")


def chart_rank_vs_model_size(points: list[ModelPoint], baseline: BaselinePoint) -> None:
    y_max = baseline.model_size_mb * 1.08
    chart = Chart(
        "Rank Fraction vs Model Size",
        "Rank fraction",
        "Model size (MB, weights)",
        0.1,
        0.52,
        0,
        y_max,
    )
    chart.axes(x_ticks=[0.125, 0.1875, 0.25, 0.375, 0.5])
    groups = grouped(points)
    chart.line(chart.left, chart.y(baseline.model_size_mb), chart.left + chart.plot_width, chart.y(baseline.model_size_mb), COLORS["baseline"], 2, "6 6")
    for family, group in groups.items():
        color = COLORS[family]
        coords = [(chart.x(point.rank), chart.y(point.model_size_mb)) for point in group]
        chart.polyline(coords, color)
        for point in group:
            chart.circle(chart.x(point.rank), chart.y(point.model_size_mb), 6, color)
    chart.text(chart.left + chart.plot_width - 4, chart.y(baseline.model_size_mb) - 8, "baseline size", size=12, anchor="end")
    chart.legend(list(groups), baseline=False)
    chart.write(OUTPUT_DIR / "07_rank_vs_model_size.svg")


def pareto_frontier(points: list[ModelPoint]) -> list[ModelPoint]:
    frontier: list[ModelPoint] = []
    best_acc = -float("inf")
    for point in sorted(points, key=lambda item: (item.model_size_mb, -item.best_acc)):
        if point.best_acc > best_acc:
            frontier.append(point)
            best_acc = point.best_acc
    return frontier


def chart_pareto(points: list[ModelPoint], baseline: BaselinePoint) -> None:
    x_max = max([baseline.model_size_mb, *[point.model_size_mb for point in points]]) * 1.08
    y_min = max(0, min(point.best_acc for point in points) - 4)
    y_max = baseline.accuracy + 2
    chart = Chart(
        "Pareto Summary: Size vs Accuracy",
        "Model size (MB, weights)",
        "Best top-1 accuracy (%)",
        0,
        x_max,
        y_min,
        y_max,
    )
    chart.axes()
    groups = grouped(points)
    for family, group in groups.items():
        color = COLORS[family]
        for point in group:
            chart.circle(chart.x(point.model_size_mb), chart.y(point.best_acc), 5.5, color)

    frontier = pareto_frontier(points)
    chart.polyline(
        [(chart.x(point.model_size_mb), chart.y(point.best_acc)) for point in frontier],
        "#0f172a",
        width=1.6,
        dash="3 7",
    )
    chart.diamond(chart.x(baseline.model_size_mb), chart.y(baseline.accuracy), 9, COLORS["baseline"])
    label_point(chart, baseline.model_size_mb, baseline.accuracy, "baseline", dx=-62, dy=-12)

    for point in points:
        key = (point.family, round(point.rank, 4))
        if key not in HIGHLIGHTS:
            continue
        chart.circle(chart.x(point.model_size_mb), chart.y(point.best_acc), 10, "none", stroke="#111827", stroke_width=2.4)
        label = f"{HIGHLIGHTS[key]} ({point.latency_ms:.1f} ms)"
        dx = 12
        dy = -12
        if point.family == "pure_svd" and abs(point.rank - 0.5) < 1e-9:
            dy = -24
        if point.family == "pure_svd" and abs(point.rank - 0.25) < 1e-9:
            dx = 16
            dy = 34
        if point.family == "hybrid_tucker":
            dx = 16
            dy = 20
        if point.family == "hybrid_tucker" and abs(point.rank - 0.25) < 1e-9:
            dx = 16
            dy = -28
        label_point(chart, point.model_size_mb, point.best_acc, label, dx=dx, dy=dy)

    chart.legend(list(groups))
    chart.text(chart.left + 10, chart.top + 24, "Black line is the observed size/accuracy frontier.", size=14, weight="700")
    chart.write(OUTPUT_DIR / "08_pareto_summary.svg")


def main() -> None:
    baseline = find_baseline()
    points = recovery_points()
    write_summary_csv(points, baseline)
    chart_accuracy_vs_size(points, baseline)
    chart_accuracy_vs_latency(points, baseline)
    chart_rank_vs_accuracy(points, baseline)
    chart_rank_vs_model_size(points, baseline)
    chart_pareto(points, baseline)
    print(f"Wrote charts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
