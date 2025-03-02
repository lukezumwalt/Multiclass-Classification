@echo off
echo Declaring Model...
python code/baseline.py

echo Ensuring model bin folder exists.
if not exist "bin" mkdir "bin"

echo -------------------------------------------------------------------
echo Training Model...
python code/train.py

echo Done!
pause