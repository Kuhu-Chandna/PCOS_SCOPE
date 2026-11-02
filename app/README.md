# PCOS SCOPE app

## Run on Windows

Open PowerShell in the `PCOS_SCOPE` project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r app\requirements.txt
python -m streamlit run app\app.py
```

If PowerShell blocks environment activation, use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r app\requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app\app.py
```

The saved model, metadata, and cleaned reference CSV must remain in their existing `models` and `data\processed` folders. The app loads these local files and does not retrain the model. Do not move `app.py` by itself.

Timeline information is held in Streamlit session memory and disappears when the session ends. Uploaded data and predictions are processed by the running app and are not written to project files. Avoid identifiable records, especially on a shared or public deployment.

The classifier is an academic research prototype. Its technical compatibility checks do not establish clinical validity, and its score is not a medical probability or diagnosis.
