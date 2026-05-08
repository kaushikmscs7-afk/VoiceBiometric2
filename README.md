Fastest start: open a terminal in this folder and run `npm run local`. It waits for the website to be ready, then opens http://127.0.0.1:8765/ automatically.

# Voice Biometric Authentication System

This project provides a voice biometric access control web app.

The browser records voice samples, FastAPI handles verification using the resemblyzer deep voice model, and SQLite stores member records and access logs.

## What it does

- Voice authentication for access control
- Administrator-managed member enrollment
- Admin dashboard for members and access logs
- Member and access record storage
- Browser-based UI served by FastAPI

## Admin passcode

Default admin passcode: 5846

## Fresh install from GitHub

Run these commands in order:

```powershell
git clone https://github.com/kaushikmscs7-afk/VoiceBiometric2.git
cd VoiceBiometric2
npm install
python -m pip install resemblyzer
npm run local
```

## How to use

1. Open http://127.0.0.1:8765/ if the browser does not open automatically.
2. Allow microphone access.
3. Unlock the dashboard with the admin passcode to enroll members.
4. Enroll each member with 6 voice samples.
5. Use Voice access to verify identity.

## One-time setup on a new machine

Install Node and Python dependencies:

```powershell
npm install
python -m pip install resemblyzer
```

After that, the app can run as long as the required packages are already installed.

## Important notes

- resemblyzer must be installed with pip for voice verification to work.
- Delete `backend/voicebiometric.sqlite3` and re-enroll all members if you reinstall or update the backend.
- The default admin passcode is 5846.

## Local files of interest

- `backend/main.py` for the API and voice matching
- `backend/static/index.html` for the UI
- `backend/static/app.js` for browser logic
- `backend/static/styles.css` for the layout
- `backend/voicebiometric.sqlite3` for local data storage
