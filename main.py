import os
import time

import cv2
from ultralytics import YOLO
import yt_dlp

from ai_analytics import AnalyticsConfig, AnalyticsEngine
from dashboard_server import start_dashboard_server

model = YOLO("best.pt")

youtube_url = "https://www.youtube.com/live/DjdUEyjx8GM?si=KK7EcIfWglSmQZVW"


ydl_opts = {
    'format': 'best',
    'quiet': True,
    'no_warnings': True,
    'extractor_args': {
        'youtube': {
            'player_client': ['android']
        }
    },
    'retries': 10,
    'sleep_interval': 2,
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(youtube_url, download=False)
    stream_url = info['url']

cap = cv2.VideoCapture(stream_url)
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 30.0

base_dir = os.path.dirname(__file__)
report_dir = os.path.join(base_dir, "reports")
live_json_path = os.path.join(report_dir, "live_metrics.json")

config = AnalyticsConfig(
    conf_threshold=0.25,
    enable_dashboard=True,
    dashboard_update_every=10,
    report_dir=report_dir,
    live_json_path=live_json_path,
    write_csv=True,
    write_json=True,
)
analytics = AnalyticsEngine(config, class_names=model.names, fps=fps)

server = None
if config.enable_dashboard:
    host = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("DASHBOARD_PORT", "8000"))
    web_root = os.path.join(base_dir, "web")
    server = start_dashboard_server(
        host=host,
        port=port,
        web_root=web_root,
        stats_path=live_json_path,
    )
    print(f"Dashboard: http://{host}:{port}")
start_time = time.time()
frame_idx = 0

cv2.namedWindow("Live Person Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live Person Detection", 1000, 600)

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        track_kwargs = {
            "persist": True,
            "tracker": "bytetrack.yaml",
            "conf": config.conf_threshold,
            "verbose": False,
        }
        if analytics.target_class_id is not None:
            track_kwargs["classes"] = [analytics.target_class_id]

        results = model.track(frame, **track_kwargs)
        kpi = analytics.process(results[0], frame_idx, time.time() - start_time)

        annotated = results[0].plot(
            conf=True,
            line_width=1,
            font_size=0.6,
        )
        annotated = analytics.draw_overlay(annotated, kpi)

        cv2.imshow("Live Person Detection", annotated)

        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    analytics.finalize()
    if server is not None:
        server.shutdown()
        server.server_close()
