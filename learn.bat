@echo off
echo Full Batch to Test if Model Learns

echo Sorting data...
python code/data_sort.py

echo Declaring Model...
python code/baseline.py

echo -------------------------------------------------------------------
echo Ensuring model bin folder exists.
if not exist "bin" mkdir "bin"

echo Executing learning report:
python code/learning_report.py

echo Done!
pause