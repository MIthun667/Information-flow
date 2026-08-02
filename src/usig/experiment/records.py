from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def checksum_record(record: dict[str, Any], checksum_field: str) -> str:
    payload = {key: value for key, value in record.items() if key != checksum_field}
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def validate_record_checksum(record: dict[str, Any], checksum_field: str) -> bool:
    return record.get(checksum_field) == checksum_record(record, checksum_field)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_json(value) + "\n")
    temporary.replace(path)


def read_valid_records(directory: Path, checksum_field: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not directory.exists():
        return records
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if validate_record_checksum(record, checksum_field):
            records[record["example_id"]] = record
    return records


def compile_jsonl(
    directory: Path,
    output: Path,
    *,
    ordered_ids: list[str],
    checksum_field: str,
) -> str:
    records = read_valid_records(directory, checksum_field)
    missing = [example_id for example_id in ordered_ids if example_id not in records]
    if missing:
        raise ValueError(f"Cannot compile incomplete records; missing {len(missing)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for example_id in ordered_ids:
            handle.write(canonical_json(records[example_id]) + "\n")
    temporary.replace(output)
    return hashlib.sha256(output.read_bytes()).hexdigest()
