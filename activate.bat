@echo off
REM Windows batch script to activate virtual environment and run commands

if "%1"=="test" goto test
if "%1"=="run" goto run
if "%1"=="server" goto server
if "%1"=="ui" goto ui

REM Default: just activate
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo Virtual environment activated. Run 'deactivate' to exit.
goto end

:test
echo Running tests...
call .venv\Scripts\activate.bat
python -m pytest tests/ -v
goto end

:run
echo Starting local development server...
call .venv\Scripts\activate.bat
python run_local.py
goto end

:server
echo Starting FastAPI server...
call .venv\Scripts\activate.bat
uvicorn src.api.server:app --reload
goto end

:ui
echo Starting Streamlit UI...
call .venv\Scripts\activate.bat
streamlit run src/api/ui.py
goto end

:end
