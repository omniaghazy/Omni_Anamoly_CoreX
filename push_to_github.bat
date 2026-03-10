@echo off
echo ===================================================
echo     Omni CoreX GitHub Repository Initializer
echo ===================================================
echo.
echo Please ensure you have Git and GitHub CLI (gh) installed.
echo If not, download Git from git-scm.com and GH CLI from cli.github.com
echo.
echo Step 1: Initialize Git
git init
echo.
echo Step 2: Add files (using .gitignore to exclude large data files)
git add .
echo.
echo Step 3: Commit initial files
git commit -m "Initial commit: Releasing Omni CoreX V1 & V2"
echo.
echo Step 4: Login to GitHub CLI (Follow prompts in browser if asked)
gh auth login --web
echo.
echo Step 5: Create Public GitHub Repository
gh repo create Omni_CoreX --public --source=. --remote=origin --push
echo.
echo Repository pushed successfully!
pause
