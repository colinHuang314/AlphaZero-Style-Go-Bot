@echo off
rem Night 2: resume runs\9x9_a for 13 hours (about 8pm to 9am). Safe to stop with Ctrl+C.
rem Keep the laptop plugged in (on battery it sleeps after 60 minutes).
cd /d %~dp0..
python -u -m gozero.loop --config configs/9x9.yaml --run runs/9x9_a --hours 13 >> runs\9x9_a_night2.log 2>&1
