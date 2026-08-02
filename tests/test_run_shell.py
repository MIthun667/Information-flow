from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
RUN = ROOT / "run.sh"


def invoke(
    command: str,
    *,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    values = os.environ.copy()
    values.update(environment or {})
    return subprocess.run(
        [str(RUN), command],
        cwd=ROOT,
        env=values,
        text=True,
        capture_output=True,
        check=False,
    )


def fake_python(tmp_path: Path, *, verify_status: int = 0, collect_status: int = 0) -> Path:
    path = tmp_path / "fake python"
    path.write_text(
        "#!/usr/bin/env bash\n"
        "case \" $* \" in\n"
        f"  *'usig.experiment.large_collection verify'*) exit {verify_status} ;;\n"
        f"  *'usig.experiment.large_collection collect'*) exit {collect_status} ;;\n"
        f"  *'usig.experiment.large_collection resume'*) exit {collect_status} ;;\n"
        "  *' -c '*) exit 1 ;;\n"
        "  *) printf 'mock python: %s\\n' \"$*\"; exit 0 ;;\n"
        "esac\n"
    )
    path.chmod(0o755)
    return path


def test_help_exits_successfully() -> None:
    result = invoke("help")
    assert result.returncode == 0
    assert "gsm8k-resume" in result.stdout
    assert "substantial GPU time" in result.stdout
    assert "clean-analysis" in result.stdout
    assert "calibration" in result.stdout


def test_unknown_command_fails() -> None:
    result = invoke("not-a-command")
    assert result.returncode == 2
    assert "Unknown command" in result.stderr


def test_check_detects_missing_manifest(tmp_path: Path) -> None:
    result = invoke(
        "check",
        environment={"DRY_RUN": "1", "MANIFEST_ROOT": str(tmp_path)},
    )
    assert result.returncode != 0
    assert "Missing required file" in result.stderr


def test_check_confirms_local_usig_import() -> None:
    result = invoke("check", environment={"DRY_RUN": "1"})
    assert result.returncode == 0
    assert "Local usig import: OK" in result.stdout


@pytest.mark.parametrize(
    ("command", "operation"),
    [
        ("gsm8k", "collect"),
        ("gsm8k-resume", "resume"),
        ("gsm8k-verify", "verify"),
    ],
)
def test_collection_resume_and_verify_command_construction(
    command: str, operation: str
) -> None:
    result = invoke(command, environment={"DRY_RUN": "1"})
    assert result.returncode == 0
    assert f"large_collection {operation}" in result.stdout
    assert "gsm8k_calibration.jsonl" in result.stdout
    assert "Qwen/Qwen2.5-1.5B-Instruct" in result.stdout


def test_verified_output_is_skipped(tmp_path: Path) -> None:
    python = fake_python(tmp_path, verify_status=0)
    output = tmp_path / "outputs"
    (output / "gsm8k_calibration").mkdir(parents=True)
    result = invoke(
        "gsm8k",
        environment={"PYTHON_BIN": str(python), "OUTPUT_ROOT": str(output)},
    )
    assert result.returncode == 0
    assert "already verified; skipping" in result.stdout


def test_partial_output_recommends_resume(tmp_path: Path) -> None:
    python = fake_python(tmp_path, verify_status=2)
    output = tmp_path / "outputs"
    (output / "gsm8k_calibration").mkdir(parents=True)
    result = invoke(
        "gsm8k",
        environment={"PYTHON_BIN": str(python), "OUTPUT_ROOT": str(output)},
    )
    assert result.returncode != 0
    assert "use the corresponding resume command" in result.stderr


def test_gsm8k_continuation_guard() -> None:
    result = invoke("all", environment={"DRY_RUN": "1"})
    assert result.returncode == 0
    assert "Stopped cleanly after GSM8K" in result.stdout
    assert "ifi_arith_source.jsonl" not in result.stdout


def test_status_output() -> None:
    result = invoke("status")
    assert result.returncode == 0
    assert "collection" in result.stdout
    assert "truthfulqa" in result.stdout


def test_analysis_rejects_incomplete_collection(tmp_path: Path) -> None:
    python = fake_python(tmp_path, verify_status=2)
    output = tmp_path / "outputs"
    (output / "squad").mkdir(parents=True)
    result = invoke(
        "analyze",
        environment={"PYTHON_BIN": str(python), "OUTPUT_ROOT": str(output)},
    )
    assert result.returncode != 0


def test_transfer_requires_all_arithmetic_domains(tmp_path: Path) -> None:
    python = fake_python(tmp_path, verify_status=2)
    result = invoke(
        "transfer",
        environment={"PYTHON_BIN": str(python), "OUTPUT_ROOT": str(tmp_path / "outputs")},
    )
    assert result.returncode != 0


def test_model_override() -> None:
    result = invoke(
        "gsm8k",
        environment={
            "DRY_RUN": "1",
            "MODEL_ID": "Qwen/Qwen2.5-3B-Instruct",
        },
    )
    assert result.returncode == 0
    assert "Qwen/Qwen2.5-3B-Instruct" in result.stdout


def test_paths_with_spaces_are_shell_quoted(tmp_path: Path) -> None:
    output = tmp_path / "output location"
    result = invoke(
        "gsm8k",
        environment={"DRY_RUN": "1", "OUTPUT_ROOT": str(output)},
    )
    assert result.returncode == 0
    assert "output\\ location" in result.stdout


def test_exit_status_is_preserved(tmp_path: Path) -> None:
    python = fake_python(tmp_path, verify_status=2, collect_status=7)
    result = invoke(
        "gsm8k",
        environment={
            "PYTHON_BIN": str(python),
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "LOG_ROOT": str(tmp_path / "logs"),
        },
    )
    assert result.returncode == 7


def test_clean_partials_requires_confirmation(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    (output / "squad").mkdir(parents=True)
    result = invoke(
        "clean-partials",
        environment={"DRY_RUN": "1", "OUTPUT_ROOT": str(output)},
    )
    assert result.returncode == 0
    assert "Inspection only" in result.stdout
    assert "--clean-confirmed" not in result.stdout


def test_clean_partials_confirmation_is_explicit(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    (output / "squad").mkdir(parents=True)
    result = invoke(
        "clean-partials",
        environment={
            "DRY_RUN": "1",
            "CONFIRM_CLEAN": "1",
            "OUTPUT_ROOT": str(output),
        },
    )
    assert result.returncode == 0
    assert "--clean-confirmed" in result.stdout


def test_frozen_artifacts_are_never_shell_delete_targets() -> None:
    source = RUN.read_text()
    assert "rm " not in source
    assert "qwen_ifi_66b0032f646fc519" in source
    assert "partial_artifacts" in source


def test_protected_valid_datasets_cannot_be_recollected() -> None:
    result = invoke("rerun", environment={"DRY_RUN": "1"})
    assert result.returncode == 2
    result = subprocess.run(
        [str(RUN), "rerun", "squad"],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Refusing to recollect" in result.stderr


def test_dataset_rerun_uses_separate_repair_root(tmp_path: Path) -> None:
    result = subprocess.run(
        [str(RUN), "rerun", "triviaqa"],
        cwd=ROOT,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "REPAIR_OUTPUT_ROOT": str(tmp_path / "repairs"),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert str(tmp_path / "repairs" / "triviaqa") in result.stdout


@pytest.mark.parametrize(
    "command",
    [
        "gsm8k-calibration-v4",
        "gsm8k-full",
        "truthfulqa-mc-calibration",
        "truthfulqa-mc",
        "trivia-extend",
    ],
)
def test_v3_gpu_commands_support_dry_run(command: str, tmp_path: Path) -> None:
    result = invoke(
        command,
        environment={
            "DRY_RUN": "1",
            "REPAIR_V3_ROOT": str(tmp_path / "v3"),
        },
    )
    assert result.returncode == 0
    assert str(tmp_path / "v3") in result.stdout


def test_invalid_repair_command_is_rejected() -> None:
    result = invoke("truthfulqa-mc-invalid")
    assert result.returncode == 2
    assert "Unknown command" in result.stderr


def test_model_calibration_dry_run_and_invalid_model(tmp_path: Path) -> None:
    valid = subprocess.run(
        [str(RUN), "model-calibration", "qwen2_5_7b"],
        cwd=ROOT,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "REPAIR_V3_ROOT": str(tmp_path / "v3"),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert valid.returncode == 0
    assert "Qwen2.5-7B-Instruct" in valid.stdout
    invalid = subprocess.run(
        [str(RUN), "model-calibration", "unsupported"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert invalid.returncode == 2
    assert "Invalid model key" in invalid.stderr


def test_v3_commands_never_target_existing_collection_root() -> None:
    source = RUN.read_text()
    assert 'repair_v3_root' in source
    assert 'truthfulqa_mc_full' in source
    assert 'destination="$(repair_v3_root)' in source
