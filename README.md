# SHINOBI VISION X

Real-time webcam hand-sign recognition with procedural anime-style VFX. Runs locally (CPU only, no accounts, no APIs).
Python 3.11 or 3.12, Windows 10/11. Stack: OpenCV, MediaPipe Hand Landmarker (Tasks API), NumPy, PySide6.

## Setup (PowerShell, from the project folder)

```powershell
py -3.12 -m venv .venv            # or: py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\download_model.py  # downloads Google's official hand_landmarker.task (~7 MB)
python scripts\download_model.py --face  # optional face model for proximity-triggered Fireball
python main.py --diagnose         # checks Python, packages, model, camera, writable folders
python main.py                    # launch   (or double-click run.bat after setup)
```
If the download fails, fetch the URL printed by the script manually and save it as `assets\models\hand_landmarker.task`.

## Using it
1. **Live Studio -> Start camera.** Mode *practice* shows landmarks, pose names, FPS, and sequence progress.
2. **Jutsu Controls -> Simple Mode** (default): close then open the right hand for Rasengan, or the left hand for Chidori. A fist-to-open transition is required; a held open palm does not repeat the effect. Move a hand near the mouth region, close it, then open it to launch a Fireball. Face proximity requires the optional face model downloaded with `python scripts\download_model.py --face`; without it, the UI reports face detection as unavailable and the other controls continue to work.
   Adjust the mouth radius and cooldown in Jutsu Controls, enable *Launch Fireball immediately on face proximity* to skip the close/open launch gesture, or turn on the optional hand-toward-camera Rasengan/Chidori motion controls.
3. Select **Classic Mode** in Jutsu Controls to use the original pose sequences and Sequence Builder. Practice, Cinematic, Sandbox, Calibration, and Performance modes remain available.
4. Default sequences (edit in **Sequence Builder**, stored in `config/sequences.json`):

| Sequence | Steps | Effect |
|---|---|---|
| Lightning Strike | fist > pointing > open_palm | Lightning |
| Energy Orb | open_palm > fist > open_palm | Energy orb on palm |
| Fire Burst | fist > peace > pointing | Fireball |
| Clone Echo | peace > fist > peace | Shadow clones (frame echo) |
| Summoning | pointing > open_palm > fist > peace > thumbs_up | Summoning circle |
| Cursed Aura | thumbs_up > fist > thumbs_up | Red aura |
| Reaper Seal | two_hand_contact > fist > two_hand_contact | Seal |
| Space Warp | pointing > thumbs_up > pointing | Spiral distortion |
| Twin Palm Awakening | both open palms, hold 1 s | Energy burst (needs 2 hands) |

Poses: `open_palm fist pointing peace thumbs_up two_hand_open two_hand_aligned two_hand_contact` plus custom poses.
`thumbs_up` requires the thumb pointing up. Two-hand poses need both hands visible.
5. **Modes:** practice, cinematic (no diagnostics), sequence, sandbox (manual effects), calibration, performance (480 px processing, half particles).
6. **Calibration:** enter a pose name, *Start*, hold the pose with one hand, press *Capture sample* 10-20 times (vary distance/angle), *Save pose*. Stored in `config/gestures.json`; usable in classic sequences.
7. Settings persist in `config/settings.json`. Screenshots/recordings go to `captures/`.

## Tests
```powershell
python -m pytest -q
```
Tests need no webcam or model (synthetic landmarks; a fake tracker is used *only* inside unit tests).

## Architecture
`vision/` camera (latest-frame thread), MediaPipe hand/optional face wrappers, per-hand transition controller, features, classifier, sequence state machine, calibration.
`vfx/` renderer + one module per effect (additive canvas -> cheap bloom -> composite; frame-space effects for clones/distortion).
`core/` pipeline, overlays (diagnostics kept separate from cinematic rendering), config, diagnostics. `app/` PySide6 UI + processing thread.
Mirroring: the frame is flipped before inference, so landmarks, effects and overlays share one coordinate system.

## Known limitations
- Shadow Clones is an approximate **frame-echo** (time-shifted, faded whole-frame copies); no person segmentation.
- Cursed Aura is anchored approximately to the upper-body area; face landmarks are used only for optional Fireball proximity, not eye effects.
- Recognition is rule-based on joint angles; expect to tune lighting/distance and use Calibration for tricky poses. Hands pressed together (seal poses) are hard for any landmark model; `two_hand_contact` uses palm-centre proximity.
- Palm orientation is exposed as a raw feature only; it is not used for classification.
- PyOpenGL is not used (no measurable benefit on CPU compositing).
