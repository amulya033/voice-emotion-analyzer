import csv
import json
import os
from datetime import datetime


class SessionRecorder:
    """Records per-hop emotion + stress snapshots for later export."""

    def __init__(self):
        self.start_time = datetime.now()
        self.records: list[dict] = []

    # ------------------------------------------------------------------ #

    def record(self, emotions: list[dict], stress: dict | None):
        self.records.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "emotions":  {e["label"]: round(e["score"], 4) for e in emotions},
            "stress":    {k: round(v, 4) for k, v in stress.items()} if stress else None,
        })

    # ------------------------------------------------------------------ #

    def save_json(self, path: str | None = None) -> str:
        path = path or self._default_path("json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "session_start": self.start_time.isoformat(timespec="seconds"),
            "duration_sec":  round(self.duration_seconds),
            "sample_count":  len(self.records),
            "records":       self.records,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path

    def save_csv(self, path: str | None = None) -> str | None:
        if not self.records:
            return None
        path = path or self._default_path("csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        emo_labels  = sorted(self.records[0]["emotions"].keys())
        stress_keys = sorted(self.records[0]["stress"].keys()) if self.records[0]["stress"] else []
        fieldnames  = ["timestamp"] + emo_labels + [f"stress_{k}" for k in stress_keys]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self.records:
                row: dict = {"timestamp": rec["timestamp"]}
                row.update(rec["emotions"])
                if rec["stress"]:
                    for k, v in rec["stress"].items():
                        row[f"stress_{k}"] = v
                writer.writerow(row)
        return path

    # ------------------------------------------------------------------ #

    @property
    def duration_seconds(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()

    @property
    def count(self) -> int:
        return len(self.records)

    def _default_path(self, ext: str) -> str:
        fname = self.start_time.strftime(f"session_%Y%m%d_%H%M%S.{ext}")
        return os.path.join("sessions", fname)
