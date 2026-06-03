from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import cv2


@dataclass
class AnalyticsConfig:
    target_class_name: str = "person"
    conf_threshold: float = 0.25
    track_history: int = 30
    speed_window: int = 200
    speed_anomaly_threshold: float = 25.0
    speed_zscore_threshold: float = 3.0
    prediction_window: int = 60
    prediction_horizon_s: float = 5.0
    report_dir: str = "reports"
    live_json_path: str = "reports/live_metrics.json"
    write_csv: bool = True
    write_json: bool = True
    enable_dashboard: bool = True
    dashboard_update_every: int = 10


class AnalyticsEngine:
    def __init__(
        self,
        config: AnalyticsConfig,
        class_names: Optional[Dict[int, str]] = None,
        fps: float = 30.0,
    ) -> None:
        self.config = config
        self.fps = fps if fps > 0 else 30.0
        self.target_class_id = _find_class_id(class_names, config.target_class_name)

        self.track_history: Dict[int, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.config.track_history)
        )
        self.speed_samples: Deque[float] = deque(maxlen=self.config.speed_window)
        self.count_series: Deque[Tuple[float, int]] = deque(maxlen=600)
        self.pred_series: Deque[Tuple[float, float]] = deque(maxlen=600)
        self.anom_series: Deque[Tuple[float, int]] = deque(maxlen=600)
        self.speed_series: Deque[Tuple[float, float]] = deque(maxlen=600)

        self.total_frames = 0
        self.total_count = 0
        self.total_speed = 0.0
        self.total_anomalies = 0
        self.max_count = 0
        self.max_speed = 0.0
        self.peak_time_s = 0.0

        self._start_time = time.time()
        self._csv_file = None
        self._csv_writer = None
        self._last_kpi: Optional[Dict[str, float]] = None
        self._live_json_path = self.config.live_json_path

        os.makedirs(self.config.report_dir, exist_ok=True)
        live_dir = os.path.dirname(self._live_json_path)
        if live_dir:
            os.makedirs(live_dir, exist_ok=True)
        if self.config.write_csv:
            csv_path = os.path.join(self.config.report_dir, _ts("metrics", "csv"))
            self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(
                [
                    "timestamp_s",
                    "frame_idx",
                    "count",
                    "avg_speed",
                    "anomaly_count",
                    "predicted_count",
                ]
            )

    def process(self, result, frame_idx: int, timestamp_s: float) -> Dict[str, float]:
        count, avg_speed, anomaly_count = self._analyze(result)
        predicted = self._predict_count(timestamp_s, count)

        self.total_frames += 1
        self.total_count += count
        self.total_speed += avg_speed
        self.total_anomalies += anomaly_count

        if count > self.max_count:
            self.max_count = count
            self.peak_time_s = timestamp_s
        if avg_speed > self.max_speed:
            self.max_speed = avg_speed

        self.count_series.append((timestamp_s, count))
        self.pred_series.append((timestamp_s, predicted))
        self.anom_series.append((timestamp_s, anomaly_count))
        self.speed_series.append((timestamp_s, avg_speed))

        if self._csv_writer:
            self._csv_writer.writerow(
                [timestamp_s, frame_idx, count, avg_speed, anomaly_count, predicted]
            )

        kpi = {
            "count": float(count),
            "avg_speed": float(avg_speed),
            "anomaly_count": float(anomaly_count),
            "predicted": float(predicted),
        }
        self._last_kpi = kpi

        if self.config.enable_dashboard and (
            frame_idx % self.config.dashboard_update_every == 0
        ):
            self._write_live_stats(frame_idx, timestamp_s, kpi, status="running")

        return kpi

    def draw_overlay(self, frame, kpi: Dict[str, float]):
        lines = [
            f"Count: {int(kpi['count'])}",
            f"Pred (5s): {kpi['predicted']:.1f}",
            f"Avg speed: {kpi['avg_speed']:.1f}",
            f"Anomaly: {int(kpi['anomaly_count'])}",
        ]

        x, y = 12, 22
        line_h = 20
        box_w = 220
        box_h = line_h * len(lines) + 10

        cv2.rectangle(frame, (8, 8), (8 + box_w, 8 + box_h), (0, 0, 0), -1)
        for i, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (x, y + i * line_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return frame

    def finalize(self) -> None:
        if self._csv_file:
            self._csv_file.close()

        duration_s = time.time() - self._start_time
        avg_count = (self.total_count / self.total_frames) if self.total_frames else 0.0
        avg_speed = (self.total_speed / self.total_frames) if self.total_frames else 0.0
        anomaly_rate = (
            (self.total_anomalies / self.total_frames) if self.total_frames else 0.0
        )

        summary = {
            "duration_s": round(duration_s, 2),
            "avg_count": round(avg_count, 2),
            "max_count": self.max_count,
            "peak_time_s": round(self.peak_time_s, 2),
            "avg_speed": round(avg_speed, 2),
            "max_speed": round(self.max_speed, 2),
            "anomaly_rate": round(anomaly_rate, 4),
            "total_frames": self.total_frames,
        }

        if self.config.write_json:
            json_path = os.path.join(self.config.report_dir, _ts("summary", "json"))
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        if self.config.enable_dashboard and self._last_kpi is not None:
            self._write_live_stats(
                self.total_frames,
                duration_s,
                self._last_kpi,
                status="finished",
            )

    def _analyze(self, result) -> Tuple[int, float, int]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return 0, 0.0, 0

        cls_list = boxes.cls.tolist() if boxes.cls is not None else []
        conf_list = boxes.conf.tolist() if boxes.conf is not None else []
        xyxy_list = boxes.xyxy.tolist() if boxes.xyxy is not None else []
        id_list = boxes.id.tolist() if boxes.id is not None else [None] * len(xyxy_list)

        count = 0
        speeds = []
        anomaly_count = 0

        for cls_id, conf, xyxy, track_id in zip(
            cls_list, conf_list, xyxy_list, id_list
        ):
            if conf < self.config.conf_threshold:
                continue
            if self.target_class_id is not None and int(cls_id) != self.target_class_id:
                continue

            count += 1

            if track_id is None:
                continue

            cx = (xyxy[0] + xyxy[2]) / 2.0
            cy = (xyxy[1] + xyxy[3]) / 2.0
            track_id_int = int(track_id)
            history = self.track_history[track_id_int]
            history.append((cx, cy))

            speed = 0.0
            if len(history) > 1:
                (x1, y1), (x2, y2) = history[-2], history[-1]
                dist = math.hypot(x2 - x1, y2 - y1)
                speed = dist
                speeds.append(speed)
                self.speed_samples.append(speed)

                if self._is_anomaly(speed):
                    anomaly_count += 1

        avg_speed = statistics.mean(speeds) if speeds else 0.0
        return count, avg_speed, anomaly_count

    def _predict_count(self, timestamp_s: float, current_count: int) -> float:
        points = list(self.count_series)
        if len(points) < 2:
            return float(current_count)

        window = points[-self.config.prediction_window :]
        t_vals = [t for t, _ in window]
        y_vals = [c for _, c in window]

        n = len(window)
        sum_t = sum(t_vals)
        sum_y = sum(y_vals)
        sum_t2 = sum(t * t for t in t_vals)
        sum_ty = sum(t * y for t, y in window)

        denom = (n * sum_t2) - (sum_t * sum_t)
        if denom == 0:
            return float(current_count)

        slope = ((n * sum_ty) - (sum_t * sum_y)) / denom
        intercept = (sum_y - slope * sum_t) / n
        future_t = timestamp_s + self.config.prediction_horizon_s
        pred = slope * future_t + intercept
        return max(0.0, pred)

    def _is_anomaly(self, speed: float) -> bool:
        if speed >= self.config.speed_anomaly_threshold:
            return True

        if len(self.speed_samples) < 5:
            return False

        mean = statistics.mean(self.speed_samples)
        std = statistics.pstdev(self.speed_samples)
        if std <= 0:
            return False

        z = (speed - mean) / std
        return z >= self.config.speed_zscore_threshold

    def _write_live_stats(
        self,
        frame_idx: int,
        timestamp_s: float,
        kpi: Dict[str, float],
        status: str,
    ) -> None:
        avg_count = (self.total_count / self.total_frames) if self.total_frames else 0.0
        avg_speed = (self.total_speed / self.total_frames) if self.total_frames else 0.0
        anomaly_rate = (
            (self.total_anomalies / self.total_frames) if self.total_frames else 0.0
        )

        series_count = list(self.count_series)[-120:]
        series_pred = list(self.pred_series)[-120:]
        series_anom = list(self.anom_series)[-120:]
        series_speed = list(self.speed_series)[-120:]

        payload = {
            "status": status,
            "timestamp_s": round(timestamp_s, 2),
            "frame_idx": frame_idx,
            "count": int(kpi.get("count", 0.0)),
            "predicted": round(float(kpi.get("predicted", 0.0)), 2),
            "avg_speed": round(float(kpi.get("avg_speed", 0.0)), 2),
            "anomaly_count": int(kpi.get("anomaly_count", 0.0)),
            "avg_count": round(avg_count, 2),
            "max_count": int(self.max_count),
            "peak_time_s": round(self.peak_time_s, 2),
            "avg_speed_total": round(avg_speed, 2),
            "max_speed": round(self.max_speed, 2),
            "anomaly_rate": round(anomaly_rate, 4),
            "total_frames": int(self.total_frames),
            "fps": round(self.fps, 2),
            "series": {
                "times": [round(t, 2) for t, _ in series_count],
                "counts": [c for _, c in series_count],
                "preds": [round(p, 2) for _, p in series_pred],
                "anomalies": [a for _, a in series_anom],
                "speeds": [round(s, 2) for _, s in series_speed],
            },
        }

        tmp_path = f"{self._live_json_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, self._live_json_path)


def _find_class_id(
    class_names: Optional[Dict[int, str]], target_name: str
) -> Optional[int]:
    if not class_names:
        return None

    target = target_name.strip().lower()
    for idx, name in class_names.items():
        if str(name).strip().lower() == target:
            return int(idx)

    for idx, name in class_names.items():
        if target in str(name).strip().lower():
            return int(idx)

    return None


def _ts(prefix: str, ext: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}.{ext}"
