
### Requirements

1. Python 3.12+
2. NVIDIA (recommended, optional)

### Setup

1. Python venv erstellen im steam-nlp Ordner: `python -m venv env`
2. venv aktivieren:
   - CMD / BAT: `env\Scripts\activate.bat`
   - Powershell: `env\Scripts\Activate.ps1`
   - Linux: `env/bin/activate`
3. Dependencies installieren:

   - Mit CUDA support:
   ```cmd
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip3 install transformers[torch]
   pip3 install pandas scikit-learn
   ```

   - Ohne CUDA support:
   ```cmd
   pip3 install transformers[torch]
   pip3 install pandas scikit-learn
   ```

4. venv deaktivieren (nur BASH / Linux) - optional: `deactivate` (unter Windows reicht es das Terminal zu schließen)


### Usage (Demo / Review)

1. venv aktivieren (falls deaktiviert), siehe Setup
2. Ausführen: `python review.py`

### Training

0. Vorhandenes Modell löschen oder umbenennen
1. venv aktivieren (falls deaktiviert), siehe Setup
2. Ausführen:
   - CMD: `python training\train.py`
   - PS1: `python .\training\train.py`
   - Linux: `python3 training/train.py`
