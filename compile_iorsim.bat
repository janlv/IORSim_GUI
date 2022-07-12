
@REM set clean=""                
@REM if "%~1"=="clean" set clean="--clean"

C:\Users\javi\AppData\Local\Programs\Python\Python39\Scripts\pyinstaller.exe %~1 --onefile  --add-data "guides;guides" --add-data "icons;icons" --add-data "dist\upgrader\upgrader.exe;." --upx-dir "C:\Users\javi\Downloads\upx-3.96-win64" .\IORSim_GUI.py