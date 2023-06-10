@echo off

echo "data creation..."
python data_creation.py

echo "model preprocess..."
python model_preprocessing.py

echo "model prep..."
python model_preparation.py

echo "model test..."
python model_testing.py

echo "DONE"
pause