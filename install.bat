@echo off
setlocal EnableDelayedExpansion
rem Install agentRW per-user to %LOCALAPPDATA%\Programs\agentRW.
rem
rem Per-user, not Program Files: that needs admin AND a system PATH edit, and
rem setx /M truncates PATH at 1024 characters - a well-known way to wreck a
rem machine. This is what VS Code and most dev CLIs do on Windows.

set "PREFIX=%LOCALAPPDATA%\Programs\agentRW"
set "HERE=%~dp0"
if "%HERE:~-1%"=="\" set "HERE=%HERE:~0,-1%"

if not exist "%HERE%\coding_agent.py" (
    echo Missing coding_agent.py - run this from the unpacked release.
    exit /b 1
)

rem Ollama is what agentRW talks to. Nothing works without it, so stop here
rem rather than install something that cannot run. Their installer, not ours.
where ollama >nul 2>&1
if errorlevel 1 (
    echo.
    echo   Ollama was not found on this machine.
    echo.
    echo   agentRW talks to a model through Ollama, so it needs to be installed
    echo   first. Get it from:
    echo.
    echo       https://ollama.com
    echo.
    echo   Install it their way, then run this again.
    echo.
    pause
    exit /b 1
)

where python >nul 2>&1
if errorlevel 1 (
    echo python not found on PATH. Install Python 3.9 or newer.
    exit /b 1
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)"
if errorlevel 1 (
    echo agentRW needs Python 3.9 or newer.
    exit /b 1
)

rem Preserve installed plugins across an upgrade.
set "KEEP="
if exist "%PREFIX%\tools" (
    set "KEEP=%TEMP%\agentRW-tools-%RANDOM%"
    echo Keeping existing plugins from %PREFIX%\tools
    xcopy /E /I /Q /Y "%PREFIX%\tools" "!KEEP!" >nul
)

echo Installing to %PREFIX%
if not exist "%PREFIX%" mkdir "%PREFIX%"
if exist "%PREFIX%\tools" rmdir /S /Q "%PREFIX%\tools"
copy /Y "%HERE%\coding_agent.py"  "%PREFIX%\" >nul
copy /Y "%HERE%\requirements.txt" "%PREFIX%\" >nul
for %%F in (README.md PLUGINS.md FUTURES.md LICENSE uninstall.bat) do (
    if exist "%HERE%\%%F" copy /Y "%HERE%\%%F" "%PREFIX%\" >nul
)
if exist "%HERE%\tools" (
    xcopy /E /I /Q /Y "%HERE%\tools" "%PREFIX%\tools" >nul
) else (
    if not exist "%PREFIX%\tools" mkdir "%PREFIX%\tools"
)
if defined KEEP (
    xcopy /E /I /Q /Y "!KEEP!" "%PREFIX%\tools" >nul
    rmdir /S /Q "!KEEP!"
)

rem The shim. No symlinks: they need admin or developer mode on Windows.
> "%PREFIX%\cagent.bat" echo @echo off
>>"%PREFIX%\cagent.bat" echo python "%PREFIX%\coding_agent.py" %%*

rem Add to the USER PATH only, read from the registry so the system half is
rem never rewritten. %PATH% cannot be used here - it is system+user expanded,
rem and writing it back would copy the system entries into the user PATH.
powershell -NoProfile -Command ^
  "$p=[Environment]::GetEnvironmentVariable('PATH','User'); if ($p -eq $null) { $p='' };" ^
  "if ($p -split ';' -notcontains '%PREFIX%') {" ^
  "  $n = if ($p -eq '') { '%PREFIX%' } else { $p.TrimEnd(';') + ';%PREFIX%' };" ^
  "  [Environment]::SetEnvironmentVariable('PATH', $n, 'User');" ^
  "  Write-Host 'Added %PREFIX% to your PATH'" ^
  "} else { Write-Host '%PREFIX% already on your PATH' }"

echo.
echo   ============================================================
echo    Install successful.  Run 'cagent' to start the harness.
echo.
echo    Open a NEW terminal first - PATH changes do not reach
echo    terminals that are already running.
echo   ============================================================
echo.
echo Python packages:
echo     pip install --user -r "%PREFIX%\requirements.txt"
echo.
echo That is the whole list. No plugins ship with the agent - each lives in its
echo own repo, and says what it needs itself in /plugins.
echo.
echo Uninstall with:  "%PREFIX%\uninstall.bat"
endlocal
