@echo off
echo ------------------------------------------------------
echo Launching YOLOv8 FastAPI App with Uvicorn...
echo ------------------------------------------------------

REM Activate your virtual environment
call venv\Scripts\activate

REM Run the FastAPI app using uvicorn
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

pause
