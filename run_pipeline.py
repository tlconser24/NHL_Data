import os
import nbformat
from nbconvert import PythonExporter
from nbconvert.preprocessors import ExecutePreprocessor

# --- CONFIG ---
NOTEBOOK_PATH = r"C:\Users\tlcon\OneDrive\Documents\GitHub\NHL_Data\notebooks\NHL_Data_project.ipynb"
OUTPUT_DIR = "data"

# --- Ensure paths exist ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load notebook ---
with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    notebook = nbformat.read(f, as_version=4)

# --- Execute notebook ---
print("🚀 Running notebook:", NOTEBOOK_PATH)
ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

try:
    ep.preprocess(notebook, {"metadata": {"path": os.getcwd()}})
    print("✅ Notebook executed successfully.")
except Exception as e:
    print("❌ Error during execution:", e)

# --- Export as Python script (optional) ---
python_exporter = PythonExporter()
source, _ = python_exporter.from_notebook_node(notebook)
with open("NHL_Data_project_converted.py", "w", encoding="utf-8") as f:
    f.write(source)

print(f"✅ Exported Python version to NHL_Data_project_converted.py")
print(f"✅ Data outputs should now exist inside the /data folder.")
