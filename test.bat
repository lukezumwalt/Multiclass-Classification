@echo off
echo Declaring Model...
python code/baseline.py

echo -------------------------------------------------------------------
echo Testing Model...
python code/test.py

echo Done!
pause