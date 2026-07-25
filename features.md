# Feature Roadmap

1. Continuous microphone listening and speech transcription.

2. Silence-based detection to determine when the user has finished speaking.

3. Direct voice conversations with Codex.

4. Written Codex output in the terminal or a Markdown interface.

5. Separate audio and written response channels:
   - Concise, conversational text for TTS.
   - Detailed Markdown, code, commands, paths, and logs for reading.

6. Microsoft Edge TTS integration to give Codex a voice.

7. Stream complete Codex sentences to TTS for lower latency.

8. Barge-in support: speaking interrupts TTS playback and optionally the active
   Codex turn.

9. Background Codex subagents that can work while the user continues talking
   to the main agent.

10. Status checks, steering, cancellation, and result collection for background
    agents.

11. Continuous capture of Zoom or other computer-output audio.

12. Capture through the existing PipeWire monitor for the USB headset.

13. Optional virtual audio routing to isolate Zoom from TTS, notifications, and
    other applications.

14. Separate audio paths for:
    - The user's microphone.
    - Meeting audio.
    - Codex TTS playback.

15. Continuous meeting transcription that never pauses while Codex performs
    analysis.

16. Timestamped storage of the complete meeting transcript.

17. A rolling meeting summary so Codex retains useful context without repeatedly
    processing the entire transcript.

18. A separate Codex conversation or agent for meeting analysis, isolated from
    the user's direct conversation.

19. Periodic meeting insights instead of sending every sentence to Codex.

20. On-demand meeting analysis through questions such as, "What do you think
    about that?"

21. Detection of meeting decisions, action items, unanswered questions,
    contradictions, and suggested follow-ups.

22. Evidence-grounded insights that refer to transcript content and abstain
    when context is insufficient.

23. Runtime operating modes:
    - `transcribe-only`
    - `insights-on`
    - `insights-off`

24. Ability to disable Codex insights while transcription continues
    uninterrupted.

25. Input-source labeling so Codex distinguishes:
    - Voice commands.
    - Typed messages.
    - Meeting transcripts.

26. Protection against treating meeting dialogue as instructions to Codex.

27. Protection against Codex TTS being transcribed back into the meeting
    transcript.

28. Handling of overlapping meeting speakers and, eventually, speaker
    identification.

29. Explicit assistant states such as `LISTENING`, `THINKING`, and `SPEAKING`.

30. Independent queues or asynchronous tasks for audio capture, transcription,
    Codex analysis, TTS, and playback.

31. Central coordination for modes, routing, cancellation, priorities, and
    clean shutdown.

32. Persistent Codex developer instructions defining the audio/display
    communication protocol.

33. Structured streamed responses using sections such as `<speak>` and
    `<display>`.

34. Privacy safeguards and participant consent for meeting transcription.

35. Continue implementing in Python initially, with an architecture that could
    later be migrated to Go if scale or reliability requirements justify it.

36. A transcript-chat view with explicit `User Voice`, `User Text`, `Them`, and
    `Codex` sources.

37. Optional selection of an audio output to transcribe as `Them`, including a
    `None` choice that disables `Them` transcription.

38. A startup response policy controlling when Codex responds:
    - After `Them`.
    - After both `User Voice` and `Them`.
    - After `User Voice`.
    - Never after voice input, while transcription continues.

39. Always-available typed input that queues `User Text` for Codex regardless of
    the selected voice-response policy.

40. Continuous context collection from `User Voice` and `Them`, including when
    those sources do not trigger an immediate Codex response.

41. A serialized Codex request queue that explicitly identifies whether each
    response is replying to `Them`, `User Voice`, or `User Text`.

42. Color-coded terminal transcripts:
    - `Them` in bright yellow.
    - `Codex` in bright green.
    - `User Voice` in bright blue.
    - `User Text` in a slightly softer, but still light, blue.
