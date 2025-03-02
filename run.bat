@echo off
echo Declaring Model...
python code/baseline.py

echo -------------------------------------------------------------------
echo Training Model...
python code/train.py

echo Done!
pause
