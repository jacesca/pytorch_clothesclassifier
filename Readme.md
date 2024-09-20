## Installing using GitHub
- Fork the project into your GitHub
- Clone it into your dektop
```
git clone https://github.com/jacesca/pytorch_examples.git
```
- Setup environment (it requires python3)
```
python -m venv venv
source venv/bin/activate  # for Unix-based system
venv\Scripts\activate  # for Windows
```
- Install requirements
```
pip install -r requirements.txt
```
- Required libraries (Review requirements.txt)
```
conda create --name deep python jupyter pandas seaborn matplotlib scikit-learn numpy python-dotenv
conda activate deep
pip install pyyaml
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  # Enables GPU
pip install ExpectException
pip install scikit-learn
pip install torchmetrics
```

