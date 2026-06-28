# event-recorder

A Python event recorder using Ultralytics YOLO and OpenCV that automatically
saves video clips when configured objects are detected.

## Setup

```bash
python3 -m venv venv
./venv/bin/python -m pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
cp config.example.yaml config.yaml
```

Running without `--config` uses `config.yaml` in this project if it exists. If
it does not exist, the app falls back to `config.example.yaml` so the script can
start from a plain checkout. For normal use, edit `config.yaml` for the camera
source, model path, target classes, and output directory.

## Run

```bash
./venv/bin/python smart_recorder.py --config config.yaml
```

You can also run the script without arguments:

```bash
./venv/bin/python smart_recorder.py
```

GUI mode is available separately:

```bash
./venv/bin/python smart_recorder_gui.py
```

In GUI mode, enable `Enable Audio` and choose a microphone to mux microphone
audio into finalized clips. If audio capture or muxing fails, the app keeps the
video-only MP4 and records the audio status in the JSON metadata.

The default configuration records MP4 clips under `recordings/YYYY-MM-DD/` when
`person` or `car` is detected. Each finalized clip gets a matching JSON metadata
file. Files still being written use a `_partial` suffix.

## Test

```bash
./venv/bin/python -m pytest
```

The unit tests cover event state transitions, class-name mapping, and the
pre-event buffer without downloading a YOLO model or accessing a real camera.
