
python -m venv env

.\env\Scripts\Activate.ps1

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install transformers[torch]
pip3 install pandas scikit-learn

