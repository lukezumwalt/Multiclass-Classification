@echo off
echo Declaring Model...
python code/baseline.py

echo -------------------------------------------------------------------
echo Training Model...
python code/train.py

echo -------------------------------------------------------------------
echo Testing Model...
python code/test.py

echo Done!
pause