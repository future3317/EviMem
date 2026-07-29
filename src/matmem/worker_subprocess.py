"""Shared subprocess machinery for secure protocol and WBM policy workers.

Domain-specific workers keep their own state models and CLI flags; this module
only owns the common pattern of launching a Python script, passing a JSON
request, validating the returned ID, and tearing the process down.
"""

from __future__ import annotations

import queue
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from .utils import _checksum


@dataclass
class _PersistentProcessState:
    """Runtime state for a JSONL worker server."""

    process: subprocess.Popen[str]
    responses: queue.Queue[str | None] = field(default_factory=queue.Queue)
    stderr: deque[str] = field(default_factory=lambda: deque(maxlen=50))


class WorkerSubprocessBase:
    """One-shot subprocess wrapper with deterministic identity checksums."""

    def __init__(
        self,
        *,
        worker_path: Path,
        selection_timeout_seconds: float,
    ) -> None:
        self.worker_path = worker_path
        self.selection_timeout_seconds = selection_timeout_seconds

    def _select_one_shot(
        self,
        command: list[str],
        request_json: str,
        *,
        error_prefix: str = "policy subprocess failed",
    ) -> str:
        result = subprocess.run(
            command,
            input=request_json,
            text=True,
            capture_output=True,
            check=False,
            timeout=self.selection_timeout_seconds,
        )
        if result.returncode != 0:
            raise RuntimeError(f"{error_prefix}: {result.stderr.strip()}")
        return result.stdout.strip()

    def _validate_returned_id(
        self,
        selected: str,
        valid_ids: set[str],
        *,
        kind: str = "candidate",
    ) -> str:
        if selected not in valid_ids:
            raise RuntimeError(f"policy subprocess returned an unknown {kind} ID")
        return selected

    def _identity_checksum(self, payload: object) -> str:
        return _checksum(payload)


class PersistentWorkerSubprocess(WorkerSubprocessBase):
    """Long-lived JSONL worker server with buffered stdout/stderr readers."""

    def __init__(
        self,
        *,
        worker_path: Path,
        selection_timeout_seconds: float,
    ) -> None:
        super().__init__(
            worker_path=worker_path,
            selection_timeout_seconds=selection_timeout_seconds,
        )
        self._persistent_state: _PersistentProcessState | None = None

    def _start_persistent(
        self,
        command: list[str],
        *,
        error_prefix: str = "persistent policy subprocess",
    ) -> None:
        if self._persistent_state is not None:
            return
        process = subprocess.Popen(
            [*command, "--serve-jsonl"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            raise RuntimeError(f"{error_prefix} pipes are unavailable")
        state = _PersistentProcessState(process=process)

        def read_stdout() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                state.responses.put(line.rstrip("\r\n"))
            state.responses.put(None)

        def read_stderr() -> None:
            assert process.stderr is not None
            for line in process.stderr:
                state.stderr.append(line.rstrip("\r\n"))

        threading.Thread(target=read_stdout, daemon=True).start()
        threading.Thread(target=read_stderr, daemon=True).start()
        self._persistent_state = state

    def _select_persistent(
        self,
        request_json: str,
        *,
        error_prefix: str = "persistent policy subprocess",
    ) -> str:
        state = self._persistent_state
        if state is None:
            raise RuntimeError(f"{error_prefix} has not been started")
        process = state.process
        if process.poll() is not None:
            self.close()
            raise RuntimeError(f"{error_prefix} exited: " + "\n".join(state.stderr))
        try:
            process.stdin.write(request_json + "\n")
            process.stdin.flush()
            selected = state.responses.get(timeout=self.selection_timeout_seconds)
        except (BrokenPipeError, queue.Empty) as exc:
            self.close()
            raise RuntimeError(
                f"{error_prefix} timed out or closed: " + "\n".join(state.stderr)
            ) from exc
        if selected is None:
            self.close()
            raise RuntimeError(f"{error_prefix} returned EOF: " + "\n".join(state.stderr))
        return selected.strip()

    def close(self) -> None:
        state = self._persistent_state
        self._persistent_state = None
        if state is None:
            return
        process = state.process
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def __enter__(self) -> PersistentWorkerSubprocess:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
