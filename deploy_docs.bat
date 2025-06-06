@echo off
setlocal enabledelayedexpansion

REM Navigate to docs directory
cd R15BSI\docs || exit /b

REM Build docs
py -m sphinx -b html source build
if errorlevel 1 exit /b

REM Create .nojekyll to prevent GitHub ignoring files starting with _
type nul > build\html\.nojekyll

REM Back to repo root
cd ..\..

REM Remove existing worktree if present
git worktree remove C:\TEMP\gh-pages 2>nul

REM Remove folder if it still exists (leftover from previous runs)
if exist C:\TEMP\gh-pages (
    rmdir /s /q C:\TEMP\gh-pages
)

REM Add worktree cleanly
git worktree add C:\TEMP\gh-pages gh-pages
if errorlevel 1 exit /b

REM Copy built docs to worktree
xcopy /e /i /y R15BSI\docs\build\html\* C:\TEMP\gh-pages\

REM Commit and push changes
cd C:\TEMP\gh-pages
git add --all
git commit -m "Update docs %DATE% %TIME%" || echo No changes to commit
git push origin gh-pages
if errorlevel 1 exit /b

REM Cleanup: remove worktree link (do not delete folder, git removes it)
cd ..
git worktree remove C:\TEMP\gh-pages

endlocal
