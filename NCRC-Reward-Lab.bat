@echo off
setlocal
cd /d "%~dp0"
title NCRC Reward Lab
python -m ncrc_lab.cli
if errorlevel 1 (
  echo.
  echo 실행 중 오류가 발생했습니다.
  pause
)
endlocal
