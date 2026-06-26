# Module rod-hockey-game 

Provide a description of the purpose of the module and any relevant information.

## Models

This module provides the following model(s):

- [`viam-rod-hockey:rod-hockey-game:rod_hockey_game`](viam-rod-hockey_rod-hockey-game_rod_hockey_game.md) - Provide a brief description of the model

## Viam App: rod-hockey-operator

This module also ships a Viam App — a browser UI that lets someone play without
opening app.viam.com or running any Python scripts. It shows connection status,
puck detection status, an auto-play toggle, and a "home all rods" button.

- Frontend: `app/dist/index.html` + `app/dist/auth.js` (no build step — the Viam
  TypeScript SDK is loaded from a CDN).
- The page only calls `do_command` on the `rod-hockey-game` service with three
  actions: `status`, `auto_play`, and `home`. All game logic stays on the robot.
- Registered via the `applications` entry in `meta.json` (entrypoint
  `app/dist/index.html`). Once uploaded it is served at
  `rod-hockey-operator_<namespace>.viamapplications.com`.

## Deploy

### 1. Required environment variables

The service reuses the existing client code in `robot/`, `main.py`, and `home.py`,
which connect to the machine using API-key credentials. Locally these come from
`.env`, but `.env` is gitignored and is NOT bundled into the module archive. So on
the deployed machine you MUST provide them as module environment variables.

In the machine config on app.viam.com, add to this module's `env`:

```json
"env": {
  "ROBOT_ADDRESS": "rig1-2270-main.<...>.viam.cloud",
  "ROBOT_API_KEY": "<api-key>",
  "ROBOT_API_KEY_ID": "<api-key-id>"
}
```

Without these the module loads, but `home`/`status`/`auto_play` fail to connect.

### 2. Build and upload

```sh
# Build the PyInstaller binary + tarball (includes meta.json, dist/module, app/dist).
./build-module.sh

# Upload the new version to the registry.
viam module upload --version <x.y.z> --platform <platform> dist/archive.tar.gz
```

After upload, configure the module on a machine, set the env vars above, and the
app appears at its viamapplications.com URL for logged-in users.

> Note: PyInstaller bundles imported Python code but not data files. The app only
> uses live vision detections today, so nothing extra is needed — but if vision
> ever loads a file by path (e.g. `zones.json`), add it to `TAR_FILES` in
> `build-module.sh`.
