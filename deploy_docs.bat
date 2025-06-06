@echo off
setlocal enabledelayedexpansion

REM Build docs
cd R15BSI\docs || exit /b
py -m sphinx -b html source build || exit /b
type nul > build\html\.nojekyll
cd ..\..

REM Remove existing worktree cleanly
git worktree remove C:\TEMP\gh-pages || echo Worktree not present, skipping removal

REM Add worktree
git worktree add C:\TEMP\gh-pages gh-pages || exit /b

REM Confirm docs built and files exist
dir R15BSI\docs\build\html\
if errorlevel 1 exit /b

REM Copy files to worktree
xcopy /e /i /y R15BSI\docs\build\html\* C:\TEMP\gh-pages\ || exit /b

REM Commit and push
cd /d C:\TEMP\gh-pages || exit /b
git add --all
git commit -m "Update docs %DATE% %TIME%" || echo No changes to commit
git push origin gh-pages || exit /b

REM Clean up - go back to repo root folder before removing worktree
cd /d C:\Users\arwilzman\OneDrive - Worcester Polytechnic Institute (wpi.edu)\git\mbl
git worktree remove C:\TEMP\gh-pages

endlocal
