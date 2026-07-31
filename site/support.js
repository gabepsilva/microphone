/* TagAlong landing demo — plain DOM, no framework. */
(() => {
  "use strict";

  const COLORS = {
    Voice: "#6ea8ff",
    Audio: "#e8b04a",
    Taga: "#d7ff3e",
    Text: "#5f86c9",
  };

  const FEATURES = [
    {
      num: "01",
      title: "Mic and meeting, kept apart",
      body: "Voice is your microphone. Audio is one application's PipeWire stream — Chromium, Zoom, whatever is playing — tapped without a virtual sink.",
    },
    {
      num: "02",
      title: "Taga is in the room",
      body: "Address it by name and it answers. Four names in the transcript: Voice, Text, Audio, and Taga. Codex is only the model underneath.",
    },
    {
      num: "03",
      title: "It never hears itself",
      body: "Taga's speech is a different PipeWire stream and is never linked into the far-end tap. That is wiring, not a filter that can miss.",
    },
    {
      num: "04",
      title: "You pick who it answers",
      body: "Voice, Audio, both, or stay silent until you type. Cycle the policy mid-session; the pause that ends a turn is editable in the sidebar.",
    },
    {
      num: "05",
      title: "Answers out loud, locally",
      body: "Piper synthesizes on this machine and reaches the first word fast. Edge is optional. Ctrl-T or the sidebar silences replies without unloading the engine.",
    },
    {
      num: "06",
      title: "Type when you can't speak",
      body: "The prompt always reaches Taga, takes slash commands, and pastes text or images. Settings you change in the sidebar write back to tagalong.yaml.",
    },
  ];

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function el(tag, attrs = {}, ...kids) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "style" && typeof v === "object") Object.assign(node.style, v);
      else if (k.startsWith("on") && typeof v === "function") node[k.toLowerCase()] = v;
      else if (v != null) node.setAttribute(k, v);
    }
    for (const kid of kids) {
      if (kid == null || kid === false) continue;
      node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return node;
  }

  function mountFeatures(root) {
    root.replaceChildren(
      ...FEATURES.map((f) =>
        el(
          "div",
          { class: "feature" },
          el("div", { class: "feature-num" }, f.num),
          el("div", { class: "feature-title" }, f.title),
          el("div", { class: "feature-body" }, f.body),
        ),
      ),
    );
  }

  function mountWave(root, count = 84) {
    const bars = Array.from({ length: count }, () => el("div", { class: "wave-bar" }));
    root.replaceChildren(...bars);
    return bars;
  }

  class Demo {
    constructor(ui) {
      this.ui = ui;
      this.alive = true;
      this.gen = 0;
      this.runGen = 0;
      this.lines = [];
      this.partial = null;
      this.tagaState = "idle";
      this.wave = Array.from({ length: ui.bars.length }, () => 0.08 + Math.random() * 0.1);
      this.meter = setInterval(() => this.tickWave(), 260);
      this.run();
    }

    destroy() {
      this.alive = false;
      this.gen++;
      clearInterval(this.meter);
    }

    get ok() {
      return this.alive && this.gen === this.runGen;
    }

    tickWave() {
      if (!this.alive) return;
      const hot = !!this.partial || this.tagaState === "streaming";
      this.wave = this.wave.map((v, i) => {
        const target = hot
          ? 0.12 + Math.random() * (0.55 + 0.4 * Math.sin(i / 6 + Date.now() / 400))
          : 0.05 + Math.random() * 0.12;
        return v + (Math.abs(target) - v) * 0.45;
      });
      this.ui.bars.forEach((bar, i) => {
        bar.style.height = `${Math.round(30 + this.wave[i] * 330)}px`;
      });
    }

    setState(patch) {
      Object.assign(this, patch);
      this.paint();
    }

    push(line) {
      const id = "l" + Math.random().toString(36).slice(2, 8);
      this.lines = [...this.lines, { id, ...line }];
      this.paint();
      return id;
    }

    patch(id, changes) {
      this.lines = this.lines.map((l) => (l.id === id ? { ...l, ...changes } : l));
      this.paint();
    }

    paint() {
      const { feed, status, stateEl } = this.ui;
      const stateColors = {
        idle: "#6f6d67",
        streaming: "#d7ff3e",
        interrupted: "#ff6a45",
        "running command": "#e8b04a",
      };
      stateEl.textContent = this.tagaState;
      stateEl.style.color = stateColors[this.tagaState] || "#8f8d86";

      feed.replaceChildren(
        ...this.lines.map((line) => {
          if (line.kind === "command") {
            return el(
              "div",
              { class: "cmd" },
              el("div", { class: "cmd-line" }, `$ ${line.cmd}`),
              ...line.output.map((out) => el("div", { class: "cmd-out" }, out)),
            );
          }
          const color = COLORS[line.label] || "#8f8d86";
          const textColor = line.label === "Taga" ? "#eaffb0" : "#f4f2ec";
          return el(
            "div",
            { class: "speech" },
            el("div", { class: "speech-label", style: { color } }, line.label),
            el(
              "div",
              { class: "speech-text", style: { color: textColor } },
              line.text,
              line.streaming ? el("span", { class: "caret" }, "  ") : null,
            ),
            line.interrupted
              ? el("div", { class: "cut" }, "⟂ cut off — you started talking")
              : null,
          );
        }),
      );

      status.replaceChildren();
      if (this.partial) {
        const color = COLORS[this.partial.speaker] || "#8f8d86";
        status.append(
          el("span", { class: "live", style: { color } }, "◉"),
          el("span", { class: "partial-label", style: { color } }, this.partial.speaker),
          el("span", { class: "partial-text" }, this.partial.text),
        );
      } else {
        status.append(
          el("span", { class: "idle-dot" }, "◌"),
          el("span", { class: "idle-text" }, "silence — mic hot, nothing pending"),
        );
      }
    }

    async partialType(speaker, text) {
      let shown = "";
      for (const w of text.split(" ")) {
        if (!this.ok) return;
        shown = shown ? `${shown} ${w}` : w;
        this.setState({ partial: { speaker, text: shown } });
        await sleep(120);
      }
    }

    async stream(id, text, cutAt) {
      let shown = "";
      for (let i = 0; i < text.length; i += 3) {
        if (!this.ok) return false;
        if (cutAt && shown.length >= cutAt) return false;
        shown += text.slice(i, i + 3);
        this.patch(id, { text: shown });
        await sleep(34);
      }
      return true;
    }

    async run() {
      this.runGen = ++this.gen;
      this.lines = [];
      this.partial = null;
      this.tagaState = "idle";
      this.paint();
      await sleep(600);
      if (!this.ok) return;

      await this.partialType("Audio", "does the migration still land in the same release if auth slips");
      if (!this.ok) return;
      this.push({
        kind: "speech",
        label: "Audio",
        text: "Does the migration still land in the same release if auth slips?",
      });
      this.setState({ partial: null });
      await sleep(650);
      if (!this.ok) return;

      this.setState({ tagaState: "streaming" });
      const a = this.push({ kind: "speech", label: "Taga", text: "", streaming: true });
      const done = await this.stream(
        a,
        "No — the migration is gated on the auth rewrite, so a slip moves both.",
        44,
      );
      if (!this.ok) return;
      this.patch(a, { streaming: false, interrupted: !done });
      this.setState({ tagaState: "interrupted" });

      await this.partialType("Voice", "Taga, ask about the staging cutover first");
      if (!this.ok) return;
      this.push({
        kind: "speech",
        label: "Voice",
        text: "Taga, ask about the staging cutover first.",
      });
      this.setState({ partial: null });
      await sleep(550);
      if (!this.ok) return;

      this.setState({ tagaState: "streaming" });
      const b = this.push({ kind: "speech", label: "Taga", text: "", streaming: true });
      await this.stream(b, "Staging is the real blocker — checking the last deploys.");
      if (!this.ok) return;
      this.patch(b, { streaming: false });
      this.setState({ tagaState: "running command" });
      await sleep(500);
      if (!this.ok) return;

      this.push({
        kind: "command",
        cmd: "git log --oneline -2 origin/staging",
        output: ["4b1c0ad  cut over staging to new auth", "9f22e71  add migration dry-run job"],
      });
      await sleep(850);
      if (!this.ok) return;

      this.setState({ tagaState: "streaming" });
      const c = this.push({ kind: "speech", label: "Taga", text: "", streaming: true });
      await this.stream(c, "Cutover landed three days ago. The date holds if both PRs merge Monday.");
      if (!this.ok) return;
      this.patch(c, { streaming: false });
      this.setState({ tagaState: "idle" });

      await sleep(3400);
      if (this.ok) this.run();
    }
  }

  function boot() {
    const features = document.getElementById("features");
    const wave = document.getElementById("waveform");
    const feed = document.getElementById("demo-feed");
    const status = document.getElementById("demo-status");
    const stateEl = document.getElementById("demo-state");
    const replay = document.getElementById("demo-replay");
    if (!features || !wave || !feed || !status || !stateEl) return;

    mountFeatures(features);
    const bars = mountWave(wave);
    const demo = new Demo({ bars, feed, status, stateEl });
    if (replay) replay.addEventListener("click", () => demo.run());
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
