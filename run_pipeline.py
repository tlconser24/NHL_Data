import os
import nbformat
from nbconvert import PythonExporter
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOK_PATH = "./NHL_Data_project.ipynb"
OUTPUT_DIR = "./app/data"
MODELS_DIR = "./app/models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    notebook = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
try:
    ep.preprocess(notebook, {"metadata": {"path": os.getcwd()}})
    print("✅ Notebook executed successfully.")
except Exception as e:
    print("❌ Error during execution:", e)
    raise
