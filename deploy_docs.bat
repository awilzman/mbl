@echo off
REM Abort on errors
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

REM Prepare worktree
git worktree add C:\temp\gh-pages gh-pages
if errorlevel 1 exit /b

REM Remove old files in worktree
rmdir /s /q C:\temp\gh-pages
mkdir C:\temp\gh-pages

REM Copy built docs to worktree
xcopy /e /i /y R15BSI\docs\build\html\* C:\temp\gh-pages\

REM Commit and push changes
cd C:\temp\gh-pages
git add --all
git commit -m "Update docs %DATE% %TIME%" || echo No changes to commit
git push origin gh-pages
if errorlevel 1 exit /b

REM Cleanup
cd ..
git worktree remove C:\temp\gh-pages

endlocal
