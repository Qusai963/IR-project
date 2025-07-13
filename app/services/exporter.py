import csv
import json
from pathlib import Path

class ExportService:
    def __init__(self, output_dir="exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def export_to_csv(self, data, filename):
        filepath = self.output_dir / f"{filename}.csv"
        with open(filepath, mode="w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "original", "processed"])
            writer.writeheader()
            writer.writerows(data)
        return str(filepath)

    def export_to_jsonl(self, data, filename):
        filepath = self.output_dir / f"{filename}.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return str(filepath)