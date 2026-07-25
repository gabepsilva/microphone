#!/usr/bin/env bash

set -u

if ! command -v pactl >/dev/null 2>&1; then
    echo "Error: pactl was not found. Install PulseAudio/PipeWire utilities first." >&2
    exit 1
fi

if ! pactl info >/dev/null 2>&1; then
    echo "Error: no PulseAudio/PipeWire audio server is available." >&2
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: ffmpeg was not found. Install it, then run this script again." >&2
    exit 1
fi

sources=()
labels=()
sinks=()
sink_labels=()

# pactl's short source format is:
# index<TAB>name<TAB>module<TAB>state<TAB>description...
while IFS=$'\t' read -r _ name _ _ description; do
    [[ -n "${name:-}" ]] || continue
    sources+=("$name")
    labels+=("${description:-$name}")
done < <(pactl list short sources)

if ((${#sources[@]} == 0)); then
    echo "No audio sources were found."
    exit 1
fi

echo "Available audio sources:"
echo
for i in "${!sources[@]}"; do
    printf '  %2d) %s\n      %s\n' "$((i + 1))" "${labels[$i]}" "${sources[$i]}"
done
echo

while :; do
    read -r -p "Select a source (1-${#sources[@]}): " choice
    if [[ "$choice" =~ ^[0-9]+$ ]] &&
       ((choice >= 1 && choice <= ${#sources[@]})); then
        break
    fi
    echo "Please enter a number from 1 to ${#sources[@]}."
done

source_name=${sources[$((choice - 1))]}

while IFS=$'\t' read -r _ name _ _ description; do
    [[ -n "${name:-}" ]] || continue
    sinks+=("$name")
    sink_labels+=("${description:-$name}")
done < <(pactl list short sinks)

if ((${#sinks[@]} == 0)); then
    echo "No audio outputs were found."
    exit 1
fi

echo
echo "Available physical and virtual audio outputs:"
echo
for i in "${!sinks[@]}"; do
    printf '  %2d) %s\n      %s\n' "$((i + 1))" "${sink_labels[$i]}" "${sinks[$i]}"
done
echo

while :; do
    read -r -p "Select the physical output for Zoom audio (1-${#sinks[@]}): " choice
    if [[ "$choice" =~ ^[0-9]+$ ]] &&
       ((choice >= 1 && choice <= ${#sinks[@]})); then
        break
    fi
    echo "Please enter a number from 1 to ${#sinks[@]}."
done

physical_sink_name=${sinks[$((choice - 1))]}
physical_sink_label=${sink_labels[$((choice - 1))]}
suffix=$$
virtual_source_name="virtual_microphone_${suffix}"
virtual_name="Virtual Mic"
virtual_sink_name="virtual_zoom_output_${suffix}"
virtual_sink_name_label="Virtual Out"

virtual_source_module=""
virtual_sink_module=""
output_loopback_module=""
mic_recording_pid=""
out_recording_pid=""

cleanup() {
    # SIGINT lets FFmpeg finalize the WAV headers cleanly.
    if [[ -n "$mic_recording_pid" ]] && kill -0 "$mic_recording_pid" 2>/dev/null; then
        kill -INT "$mic_recording_pid" 2>/dev/null || true
        wait "$mic_recording_pid" 2>/dev/null || true
    fi
    if [[ -n "$out_recording_pid" ]] && kill -0 "$out_recording_pid" 2>/dev/null; then
        kill -INT "$out_recording_pid" 2>/dev/null || true
        wait "$out_recording_pid" 2>/dev/null || true
    fi
    [[ -n "$output_loopback_module" ]] && \
        pactl unload-module "$output_loopback_module" >/dev/null 2>&1 || true
    [[ -n "$virtual_sink_module" ]] && \
        pactl unload-module "$virtual_sink_module" >/dev/null 2>&1 || true
    [[ -n "$virtual_source_module" ]] && \
        pactl unload-module "$virtual_source_module" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# Create a genuine Audio/Source rather than a sink monitor. Browsers and
# conferencing applications therefore see it directly as a microphone.
virtual_source_module=$(pactl load-module module-virtual-source \
    "master=${source_name}" \
    "source_name=${virtual_source_name}" \
    "source_properties=device.description='${virtual_name}'") || {
    echo "Error: could not create the virtual microphone." >&2
    echo "Your audio server may not provide module-virtual-source." >&2
    exit 1
}

# Create a virtual output. Zoom sends audio here; the loopback below forwards
# it to the selected physical output. Its monitor is also available to apps.
virtual_sink_module=$(pactl load-module module-null-sink \
    "sink_name=${virtual_sink_name}" \
    "sink_properties=device.description='${virtual_sink_name_label}'") || {
    echo "Error: could not create the virtual Zoom output." >&2
    exit 1
}

output_loopback_module=$(pactl load-module module-loopback \
    "source=${virtual_sink_name}.monitor" \
    "sink=${physical_sink_name}" \
    "source_dont_move=true" \
    "sink_dont_move=true") || {
    echo "Error: could not route the virtual output to the physical output." >&2
    exit 1
}

timestamp=$(date +%Y%m%d-%H%M%S)
mic_file="virtual-mic-${timestamp}.wav"
out_file="virtual-out-${timestamp}.wav"

ffmpeg -hide_banner -loglevel error \
    -f pulse -i "$virtual_source_name" \
    -ac 1 -ar 16000 -codec:a pcm_s16le "$mic_file" &
mic_recording_pid=$!

ffmpeg -hide_banner -loglevel error \
    -f pulse -i "${virtual_sink_name}.monitor" \
    -ac 1 -ar 16000 -codec:a pcm_s16le "$out_file" &
out_recording_pid=$!

sleep 1
if ! kill -0 "$mic_recording_pid" 2>/dev/null ||
   ! kill -0 "$out_recording_pid" 2>/dev/null; then
    echo "Error: one of the recordings could not be started." >&2
    exit 1
fi

echo
echo "Virtual microphone created:"
echo "  ${virtual_name}"
echo
echo "Virtual Zoom output created:"
echo "  ${virtual_sink_name_label}"
echo "  Zoom should send audio to this output."
echo "  Your app should listen to: Monitor of ${virtual_sink_name_label}"
echo "  Audio is also routed to: ${physical_sink_label}"
echo
echo "Recording microphone to: ${mic_file}"
echo "Recording Zoom output to: ${out_file}"
echo
read -r -p "Press Enter to stop recording and remove the virtual devices... " _
echo "Stopping recordings and removing virtual devices."
