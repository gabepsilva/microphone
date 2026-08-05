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

Agent `message.send` is refused as inapplicable until the `Agent` transcript
source lands as its own evaluated change. That path has no live client yet, so
it is not a user-visible deviation from today's session.
