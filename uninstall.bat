@echo off
setlocal EnableDelayedExpansion
rem Remove agentRW from %LOCALAPPDATA%\Programs\agentRW and from the user PATH.

set "PREFIX=%LOCALAPPDATA%\Programs\agentRW"

if not exist "%PREFIX%" (
    echo agentRW is not installed at %PREFIX%.
    exit /b 0
)

rem Say what goes before it goes. Installed plugins are deleted with everything
rem else and are not backed up anywhere.
if exist "%PREFIX%\tools" (
    set "FOUND="
    for /D %%O in ("%PREFIX%\tools\*") do (
        for /D %%P in ("%%O\*") do (
            if not defined FOUND echo These installed plugins will be deleted:
            set "FOUND=1"
            echo     %%~nxO/%%~nxP
        )
    )
    if defined FOUND echo.
)

set /P "ANS=Remove %PREFIX% and its PATH entry? [y/N] "
if /I not "%ANS%"=="y" if /I not "%ANS%"=="yes" (
    echo Nothing removed.
    exit /b 0
)

rem USER PATH only, read from the registry - never touch the system half.
powershell -NoProfile -Command ^
  "$p=[Environment]::GetEnvironmentVariable('PATH','User'); if ($p -ne $null) {" ^
  "  $n=($p -split ';' ^| Where-Object { $_ -ne '%PREFIX%' -and $_ -ne '' }) -join ';';" ^
  "  [Environment]::SetEnvironmentVariable('PATH', $n, 'User');" ^
  "  Write-Host 'Removed %PREFIX% from your PATH'" ^
  "}"

rem Cannot delete the directory this script is running from, so hand off.
if /I "%~dp0"=="%PREFIX%\" (
    start "" /B cmd /C "timeout /T 1 >nul & rmdir /S /Q ""%PREFIX%"" & echo Removed %PREFIX%"
) else (
    rmdir /S /Q "%PREFIX%"
    echo Removed %PREFIX%
)

echo.
echo Left alone, because it is yours and not ours to delete:
echo     %%APPDATA%%\coding_agent\   saved model and flags
echo     any DEBT.md or *.bak files in your projects
echo.
echo Close and reopen your terminal for the PATH change to take effect.
endlocal
