# Intentional behavior changes

Read this when a parity test fails during the issue #81 refactor, or before
adding a behavior change to it. Nothing else needs it.

The refactor's safety net is the high-level TUI journeys: they are the oracle
that says the runtime still behaves as it did. An oracle cannot tell preserved
behavior from a preserved bug, so a defect fixed inside the refactor makes the
tests that were meant to prove "nothing changed" either fail for a good reason
or turn out never to have asserted it.

This file is the boundary. Every deliberate change of behavior is listed here
with the commit that made it and the test that pins it. **A behavior change
that is not on this list is a regression**, and that is what makes the parity
claim falsifiable instead of decorative.

Recording a change here does not license it. Each entry was a defect fixed
before the baseline was frozen, on its own commit, with a regression test
verified by `make verify-regression` to fail without the fix.

## Deviations

### 1. A device selection that fails to open is no longer persisted

- **Commit:** `9abba80` (PR #83), milestone 1.
- **Was:** `select` returned `True` before the device work happened, so
  `remembering()` wrote the choice to the startup config immediately. A
  microphone or far end that then failed to open left the file naming a device
  the session never reached, and the next session started somewhere this one
  never went — exactly what `remembering()`'s own docstring forbids.
- **Is:** configuration is written from the applied outcome only. A failed
  selection writes nothing, and a superseded one never writes at all.
- **Pinned by:** `test_a_failed_microphone_selection_is_not_remembered`,
  `test_failed_application_selection_is_not_remembered`,
  `test_an_aba_selection_only_completes_the_newest_request`.

### 2. A far end selected while muted now opens muted

- **Commit:** `2d16b20` (PR #83), milestone 1.
- **Was:** `AudioChannel._open` bound the mute hook but never replayed
  `state.audio.muted`. Muting the far end with no application selected and
  then selecting one left the sidebar showing muted while the channel
  transcribed. The microphone path had always replayed its desired mute.
- **Is:** the far end replays desired mute on open, symmetrically with the
  microphone.
- **Pinned by:** `test_an_application_opened_while_muted_starts_muted`.

## Milestone 2

Milestone 2 (`tagalong/control/`) adds no deviation. The package is a new,
self-contained application core with its own tests; no existing runtime path
dispatches through it yet, so no observable session behavior changes. The
first entries from wiring a client to it belong to milestone 3, where the TUI
slice is converted.

## Milestone 3

Milestone 3 converts `message.send`, `tts.set_enabled`, `session.interrupt`,
and `session.new` onto the controller. Observable TUI journeys are unchanged:
typed text still ingests as `Text` and always requests a reply, a refused TTS
toggle still leaves the sidebar as it was, interrupt still flushes pending
stream text before cutting the turn, and `/new` still clears the transcript
only after a new Codex thread starts.

Canonical `AppState` in this slice carried only `tts_enabled`. Seeding the
rest waited for milestone 6, which registers the handlers that keep those
fields true.

Agent `message.send` is refused as inapplicable until the `Agent` transcript
source lands as its own evaluated change. That path has no live client yet, so
it is not a user-visible deviation from today's session.

## Milestone 4

Milestone 4 makes `/new` an adapter over `session.new` and `/help` a rendering
of `commands.list`. Observable TUI journeys are unchanged: `/new` still starts
a fresh session, `/help` still lists `/new` and `/help` with the same copy,
and `/help <name>` still names aliases or reports an unknown command.

`commands.list` is a query that returns structured rows (name, aliases,
summary, target action). It is not a catalog action and not
`command.invoke`. `/new`'s summary is `session.new`'s summary, so the palette
and an agent tool schema cannot drift. `/help` itself has no typed equivalent;
it only renders that listing.

## Milestone 5

Milestone 5 adds the local transport and the first second-writer path. The TUI
subscribes to controller events so a remote `tts.set_enabled` updates the
sidebar. Local toggles still note "tts on/off" themselves; a remote change
updates the sidebar without a duplicate note. Journey tests that never attach
a controller are unchanged.

The Unix socket lives under ``$XDG_RUNTIME_DIR/tagalong`` (mode ``0700``,
socket ``0600``, ``SO_PEERCRED`` same-uid only). There is no ``/tmp``
fallback: if the runtime dir is unset the TUI still runs and remote clients
cannot attach. MCP tools are generated from the static catalog; Electron is a
preload-isolated client of the same socket, not a second Python runtime.

Socket callers are agents. The ``client`` string on ``initialize`` is a label
for the actor id, not a way to mint ``ActorKind.HUMAN``. The person at the
TUI is still ``local_user`` in-process; a same-uid socket process that claims
to be Electron cannot ingest ``Text``.

## Milestone 6

Milestone 6 wires the settings and audio catalog actions and seeds every
``AppState`` field from the live session. Observable TUI journeys are
unchanged: sidebar picks still accept or refuse through the same hooks, a
clamped silence window is still what is saved, mute still reaches capture,
and a failed or superseded device selection still writes nothing.

``snapshot()`` now describes the session rather than dataclass defaults.
Desired microphone and far-end names are seeded at start; effective names
arrive when the reconcilers settle, the same accepted-then-settle shape
``session.new`` already uses. Mute, policy, provider, model, reasoning, and
silence stay synchronous.

Capture channels no longer rebind ``on_mute`` / ``on_audio_mute``. Those hooks
dispatch; the channel exposes ``set_muted`` and replays ``tui.state`` mute on
open. A session with no far end still accepts ``audio_stream.set_muted`` as
desired state. That is not a user-visible change: the checkbox still works
when a channel exists, and still records mute when it does not.

Settings persistence is on the handler, matching audio selections. A socket
``codex.set_model`` writes the startup file the same way a sidebar pick does.
Pinned by ``test_a_controller_settings_change_is_remembered`` and
``test_settings_persistence_runs_from_the_handler_not_the_hook``.

Re-enabling TTS when the sidebar speech picker leaves ``No voice reply`` is
still a TUI composition: the picker then dispatches ``tts.set_enabled``.
``tts.set_provider`` itself does not unmute, for any client.
