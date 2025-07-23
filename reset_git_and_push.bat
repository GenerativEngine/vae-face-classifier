@echo off
echo ------------------------------------------------------
echo 🧠 Resetting Git for VAE + SVM Face Recognition Project
echo ------------------------------------------------------

REM Navigate to the current directory (optional safety)
cd /d %~dp0

REM Delete old .git directory if exists
IF EXIST .git (
    rmdir /s /q .git
    echo ✅ Old Git repo removed
)

REM Initialize new Git repository
git init
echo ✅ Initialized new Git repo

REM Add .gitignore for common unwanted files
IF NOT EXIST .gitignore (
    echo venv/> .gitignore
    echo __pycache__>> .gitignore
    echo *.pyc>> .gitignore
    echo *.pkl>> .gitignore
    echo *.h5>> .gitignore
    echo *.ipynb_checkpoints/>> .gitignore
    echo .vscode/>> .gitignore
    echo .idea/>> .gitignore
    echo ✅ Created .gitignore
)

REM Add and commit files
git add .
git commit -m "Initial commit for VAE + SVM face recognition project"
git branch -M main

REM --- 🔧 EDIT THIS LINE: put your real GitHub repo URL here ---
git remote add origin https://github.com/GenerativEngine/vae-face-recognition.git

REM Push to GitHub
git push -u origin main

echo ------------------------------------------------------
echo ✅ VAE repo pushed to GitHub successfully!
echo ------------------------------------------------------
pause
