@echo off
chcp 65001
set TORCH_DEVICE=xpu

echo ========================================
echo Thesis experiment started
echo ========================================
echo.

echo [1/7] Train LSTM encoder
python train_lstm.py > log_01_train_lstm.txt 2>&1
if errorlevel 1 goto error

echo [2/7] Run pretest
python main_pretest.py > log_02_pretest.txt 2>&1
if errorlevel 1 goto error

echo [3/7] Run OLS
python main_ols.py > log_03_ols.txt 2>&1
if errorlevel 1 goto error

echo [4/7] Run Ridge and Lasso
python main_ridge.py > log_04_ridge_lasso.txt 2>&1
if errorlevel 1 goto error

echo [5/7] Run Random Forest
python main_rf.py > log_05_rf.txt 2>&1
if errorlevel 1 goto error

echo [6/7] Run XGBoost
python main_xgboost.py > log_06_xgboost.txt 2>&1
if errorlevel 1 goto error

echo [7/7] Run statistics
python main_stat.py > log_07_stat.txt 2>&1
if errorlevel 1 goto error

echo.
echo ========================================
echo All thesis scripts finished successfully
echo ========================================
pause
exit /b 0

:error
echo.
echo ========================================
echo ERROR: One script failed.
echo Check the latest log file.
echo ========================================
pause
exit /b 1