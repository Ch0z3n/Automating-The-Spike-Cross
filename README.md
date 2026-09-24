# The Spike Cross training bot prototype

This is an **offline/training-mode** screen-reading prototype for the Windows
Steam release of The Spike Cross. It captures a fixed part of the screen,
tracks the ball and the controlled player by colour, predicts the ball's short
term path, and chooses simple movement/jump/spike actions.

It starts in observation-only mode. No input is sent until you explicitly arm
it with `F10`.

## Quick start

1. Open The Spike Cross in **windowed or borderless** mode and enter Training.
2. Create the local environment (once):

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Copy `config.example.json` to `config.json` and set `capture` to the game
   court's screen coordinates (`left`, `top`, `width`, `height`).
4. Run:

   ```powershell
   $env:PYTHONPATH = "$PWD\src"
   .\.venv\Scripts\python.exe -m spike_bot --config config.json
   ```

5. Watch the console until tracking looks stable. Press `F10` to arm/disarm
   keyboard control. Press `F12` to stop.

`F8`/`F9` colour sampling remains available as a fallback detector, but the
default screen detector uses the cyan controlled-player marker and ball shape.

## Safety and limitations

- Use this only in Training, story, or another mode where automation is
  permitted. Do not use it to gain an advantage over another person.
- Keep the game focused after arming; input is sent to the active window.
- The first controller is intentionally conservative and assumes the player is
  on the left side of the court. Key bindings and timings are configurable.
- Lighting/effects can confuse colour tracking. Sampling a distinctive part of
  the ball and player generally works best.

## Configuration

`capture` is an absolute screen rectangle. `court` values are fractions inside
that rectangle. `keys` must match the game's keyboard bindings. The example
uses arrow keys plus `z` for jump/spike.

Useful tuning values:

- `vision.*_tolerance`: allowed RGB colour distance.
- `vision.*_min_pixels`: rejects tiny effects and particles.
- `vision.ball_roi`: fractional left/right/top/bottom bounds for the current
  drill's ball lane. Widen this later for full matches.
- `controller.target_lead_seconds`: how far ahead to predict the ball.
- `controller.action_cooldown_seconds`: prevents button spam.
- `controller.attack_height_ratio`: vertical threshold for jumping/attacking.

## Tests

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Unattended calibration

Open the Spike Drill player screen or its Training Complete screen, then
double-click `run_autotune.cmd`. The tuner starts games, plays each round,
restarts, and rotates timing candidates automatically. It stores persistent
results in `autotune_state.json`, so stopping and resuming does not discard
completed rounds. Press `F12` while the game is focused to stop safely.
