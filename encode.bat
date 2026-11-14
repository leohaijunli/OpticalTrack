@echo off
rem ---------------------------------------------------------------------------
rem  encode.bat - turn the rendered mocap-camera frames into delivery clips.
rem
rem  Input : out\frames\CAM01_vis\f0000.png ... f0599.png   (1280x800, mono)
rem          ... CAM06_vis
rem  Output: out\video\01_cam01_visible_20s.mp4               (clean plate)
rem          out\video\01_cam01_visible_20s_hud.mp4         (frame/time overlay)
rem
rem  Frame f of every camera is rendered from the same scene at the same
rem  animation frame, so the six clips are synchronised by construction:
rem  f0000.png of CAM01..CAM06 show the same instant, t = f / 30 s.
rem ---------------------------------------------------------------------------
setlocal enabledelayedexpansion
set ROOT=%~dp0
set F=%ROOT%out\frames
set V=%ROOT%out\video
if not exist "%V%" mkdir "%V%"
set FONT=C\:/Windows/Fonts/consola.ttf
set FONTB=C\:/Windows/Fonts/consolab.ttf

set START=0
set COUNT=600
set FPS=30

for %%C in (01 02 03 04 05 06) do (
  echo [encode] CAM%%C

  rem ---- clean plate: mono OV9281 view, nothing burned in --------------------
  ffmpeg -y -start_number %START% -framerate %FPS% -i "%F%\CAM%%C_vis\f%%04d.png" -frames:v %COUNT% ^
   -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -movflags +faststart ^
   "%V%\%%C_cam%%C_visible_20s.mp4"

  rem ---- same clip with camera id / frame number / timecode -----------------
  ffmpeg -y -start_number %START% -framerate %FPS% -i "%F%\CAM%%C_vis\f%%04d.png" -frames:v %COUNT% ^
   -vf "drawtext=fontfile='%FONTB%':text='CAM %%C  |  OV9281 mono  1280x800':x=36:y=28:fontsize=22:fontcolor=white@0.92, drawtext=fontfile='%FONT%':text='M27280M07S 2.8 mm  |  HFOV 68.9 deg  |  global shutter':x=36:y=62:fontsize=15:fontcolor=white@0.58, drawtext=fontfile='%FONT%':text='f %{eif\:n\:d\:4}   t %{pts\:hms}':x=w-tw-36:y=32:fontsize=17:fontcolor=white@0.80, drawtext=fontfile='%FONT%':text='Crazyflie 2.1  -  3 x 20 mm markers':x=36:y=h-46:fontsize=15:fontcolor=white@0.58, drawtext=fontfile='%FONT%':text='20 s @ 30 fps':x=w-tw-36:y=h-46:fontsize=15:fontcolor=white@0.58" ^
   -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -movflags +faststart ^
   "%V%\%%C_cam%%C_visible_20s_hud.mp4"
)

echo done.  -> %V%
exit /b 0
