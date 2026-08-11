"""Recognizing this program's own helper processes, and sweeping up leftovers.

``/proc`` is stood up as a directory of files rather than mocked, because what
these read is a file format: a NUL-separated environment block. A fake that
returned dictionaries would agree with any parser, including a wrong one.
"""

from __future__ import annotations

import os
import signal
import sys

import pytest

import tagalong.session as session
from tagalong import session_darwin, session_proc
from tagalong.session import (
    SESSION_MARKER,
    is_running,
    orphans,
    session_of,
    started_here,
    sweep_orphans,
    tagged_environment,
)


@pytest.fixture(autouse=True)
def use_proc_session_backend(monkeypatch):
    """Run the Linux ``/proc`` tests against their explicit backend."""
    monkeypatch.setattr(session, "_DEFAULT_PLATFORM", "linux")


def test_the_session_selector_is_injectable():
    assert session.default_session_backend("linux") is session_proc
    assert session.default_session_backend("darwin") is session_darwin


def test_darwin_ancestry_identifies_a_child_without_proc_environment():
    parents = {42: 7, 7: 1}

    assert (
        session_darwin.started_here(42, own_pid=1, parent=lambda pid: parents.get(pid))
        is True
    )
    assert (
        session_darwin.session_of(42, own_pid=1, parent=lambda pid: parents.get(pid))
        == 1
    )
    assert session_darwin.started_here(43, own_pid=1, parent=lambda _pid: None) is False
    assert session_darwin.orphans() == []
    assert session_darwin.sweep_orphans() == 0


@pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS libproc (#152)")
def test_darwin_parent_lookup_uses_real_libproc_for_this_process():
    assert session_darwin.parent_process(os.getpid()) is not None
    assert session_darwin.parent_process(0) is None


def test_darwin_libproc_configures_the_native_query_shape(monkeypatch):
    class ProcPidInfo:
        argtypes = None
        restype = None

        def __call__(self, _pid, _kind, _flavor, data, _size):
            data[
                session_darwin.PROC_BSDINFO_PPID_OFFSET : session_darwin.PROC_BSDINFO_PPID_OFFSET
                + 4
            ] = (7).to_bytes(4, "little")
            return session_darwin.PROC_BSDINFO_PPID_OFFSET + 5

    class Library:
        proc_pidinfo = ProcPidInfo()

    library = Library()
    monkeypatch.setattr(session_darwin.sys, "platform", "darwin")
    monkeypatch.setattr(session_darwin.ctypes, "CDLL", lambda _path: library)

    assert session_darwin.parent_process(42) == 7
    assert library.proc_pidinfo.restype is session_darwin.ctypes.c_int


def test_darwin_session_of_defaults_to_this_process():
    assert (
        session_darwin.session_of(
            os.getpid(), parent=lambda pid: os.getppid() if pid == os.getpid() else None
        )
        == os.getpid()
    )


def test_darwin_started_here_rejects_invalid_pids():
    assert session_darwin.started_here("42") is False
    assert session_darwin.started_here(0) is False


def test_darwin_libproc_and_parent_failures_are_closed(monkeypatch):
    monkeypatch.setattr(session_darwin.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="libproc"):
        session_darwin._libproc()

    class Unavailable:
        def proc_pidinfo(self, *_args):
            raise OSError("gone")

    class Short:
        def proc_pidinfo(self, *_args):
            return session_darwin.PROC_BSDINFO_PPID_OFFSET + 4

    assert session_darwin.parent_process(42, library=Unavailable()) is None
    assert session_darwin.parent_process(42, library=Short()) is None


def test_darwin_ancestry_stops_at_init_when_it_is_not_the_session():
    parents = {42: 7, 7: 1}

    assert (
        session_darwin.started_here(42, own_pid=99, parent=lambda pid: parents.get(pid))
        is False
    )

    parents = {42: 43, 43: 44, 44: 45}
    assert (
        session_darwin.started_here(
            42, own_pid=99, parent=lambda pid: parents.get(pid), limit=2
        )
        is False
    )


def test_the_proc_backend_reports_a_missing_process():
    def refuse(_pid, _signal):
        raise ProcessLookupError

    assert session_proc.is_running(4242, kill=refuse) is False
    assert session_proc.is_running(os.getpid()) is True


def test_the_common_session_port_accepts_default_backend_arguments():
    assert session.orphans(own_pid=os.getpid(), running=lambda _pid: True) == []


def fake_proc(tmp_path, processes):
    """Build a /proc holding one environ file per process."""
    for pid, environment in processes.items():
        directory = tmp_path / str(pid)
        directory.mkdir()
        block = b"".join(
            f"{name}={value}".encode() + b"\0" for name, value in environment.items()
        )
        (directory / "environ").write_bytes(block)
    return str(tmp_path)


def test_a_started_process_carries_the_session_that_started_it() -> None:
    environment = tagged_environment({"PATH": "/usr/bin"})

    assert environment[SESSION_MARKER] == str(os.getpid())
    assert environment["PATH"] == "/usr/bin"


def test_the_tag_names_the_session_rather_than_the_process_it_lands_in() -> None:
    """The sweep matches a helper to its owner, not to itself."""
    assert tagged_environment({}, pid=4242)[SESSION_MARKER] == "4242"


def test_a_tagged_process_reports_the_session_that_started_it(tmp_path) -> None:
    proc = fake_proc(tmp_path, {7: {"PATH": "/usr/bin", SESSION_MARKER: "99"}})

    assert session_of(7, proc=proc) == 99


def test_an_untagged_process_belongs_to_nobody_here(tmp_path) -> None:
    proc = fake_proc(tmp_path, {7: {"PATH": "/usr/bin"}})

    assert session_of(7, proc=proc) is None
    assert started_here(7, proc=proc) is False


def test_a_process_that_cannot_be_read_belongs_to_nobody_here(tmp_path) -> None:
    """Another user's process is not this program's to claim or to kill."""
    assert session_of(7, proc=str(tmp_path)) is None


def test_a_tag_that_is_not_a_number_is_no_tag_at_all(tmp_path) -> None:
    proc = fake_proc(tmp_path, {7: {SESSION_MARKER: "not-a-pid"}})

    assert session_of(7, proc=proc) is None


def test_a_variable_that_merely_starts_the_same_is_not_the_tag(tmp_path) -> None:
    proc = fake_proc(tmp_path, {7: {f"{SESSION_MARKER}_EXTRA": "99"}})

    assert session_of(7, proc=proc) is None


def test_a_helper_of_this_program_is_recognized_however_it_was_reparented(
    tmp_path,
) -> None:
    """An orphan's parent is init, so ancestry cannot answer this."""
    proc = fake_proc(tmp_path, {7: {SESSION_MARKER: "99"}})

    assert started_here(7, proc=proc) is True


def test_a_live_process_is_running() -> None:
    assert is_running(os.getpid()) is True


def test_a_process_that_is_gone_is_not_running() -> None:
    def refuse(_pid, _signal):
        raise ProcessLookupError

    assert is_running(4242, kill=refuse) is False


def test_a_helper_whose_session_is_gone_is_an_orphan(tmp_path) -> None:
    proc = fake_proc(
        tmp_path,
        {
            11: {SESSION_MARKER: "900"},
            12: {"PATH": "/usr/bin"},
        },
    )

    assert orphans(proc=proc, own_pid=1, running=lambda _pid: False) == [11]


def test_a_helper_of_a_living_session_is_left_alone(tmp_path) -> None:
    """Another session's helpers are its business, not this one's."""
    proc = fake_proc(tmp_path, {11: {SESSION_MARKER: "900"}})

    assert orphans(proc=proc, own_pid=1, running=lambda _pid: True) == []


def test_this_sessions_own_helpers_are_never_orphans(tmp_path) -> None:
    """The sweep runs while this session is starting the very helpers it lists."""
    proc = fake_proc(tmp_path, {11: {SESSION_MARKER: "42"}})

    assert orphans(proc=proc, own_pid=42, running=lambda _pid: False) == []


def test_entries_that_are_not_processes_are_skipped(tmp_path) -> None:
    (tmp_path / "self").mkdir()
    (tmp_path / "cpuinfo").write_text("", encoding="utf-8")
    proc = fake_proc(tmp_path, {11: {SESSION_MARKER: "900"}})

    assert orphans(proc=proc, own_pid=1, running=lambda _pid: False) == [11]


def test_an_unreadable_proc_yields_no_orphans() -> None:
    assert orphans(proc="/nonexistent", own_pid=1) == []


def test_sweeping_ends_the_orphans_and_counts_them(tmp_path) -> None:
    proc = fake_proc(
        tmp_path,
        {11: {SESSION_MARKER: "900"}, 12: {SESSION_MARKER: "901"}},
    )
    signalled: list[tuple[int, int]] = []

    swept = sweep_orphans(
        proc=proc,
        own_pid=1,
        running=lambda _pid: False,
        kill=lambda pid, sig: signalled.append((pid, sig)),
    )

    assert swept == 2
    assert sorted(signalled) == [(11, signal.SIGTERM), (12, signal.SIGTERM)]


def test_an_orphan_that_exits_first_is_not_counted(tmp_path) -> None:
    """Being gone already is the outcome the sweep wanted."""
    proc = fake_proc(tmp_path, {11: {SESSION_MARKER: "900"}})

    def refuse(_pid, _sig):
        raise ProcessLookupError

    swept = sweep_orphans(proc=proc, own_pid=1, running=lambda _pid: False, kill=refuse)

    assert swept == 0


@pytest.mark.parametrize("marker", ["", "="])
def test_a_malformed_environment_entry_is_survived(tmp_path, marker) -> None:
    directory = tmp_path / "7"
    directory.mkdir()
    (directory / "environ").write_bytes(marker.encode() + b"\0")

    assert session_of(7, proc=str(tmp_path)) is None
