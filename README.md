# Expression Detection Project (Scaffold)

This repository contains a scaffold for a webcam-based face detection, tracking, and expression classification project.

Quick start

1. Create a Python virtual environment and install requirements:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Replace the placeholder model in `models/svm_expression_model.pkl` with a trained sklearn joblib dump.
3. Replace the Haar cascade file in `haarcascades/haarcascade_frontalface_default.xml` with the official OpenCV xml.
4. Run the demo:

```powershell
python src\main.py
```

Controls

- `q` to quit
- `s` to save current frame to `saved_frames/`

Files of interest

- `src/main.py`: runner
- `src/face_detection.py`: Haar-cascade detection
- `src/face_tracking.py`: simple centroid tracker
- `src/feature_extraction.py`: feature stub
- `src/expression_classifier.py`: model loader & predictor
- `requirements.txt`: Python deps

Notes

This is a scaffold with placeholder files. Replace placeholders with trained models, real images and documentation as needed.

You **do NOT** need to reinstall anything every time.
You're only supposed to do **ONE thing** each time you open the project:

---

# ✅ What you do **EVERY TIME**

(from your project folder)

```
.\\.venv\\Scripts\\Activate.ps1
```

That’s it.

After activation, you can run:

```
python -m src.main
```

or

```
python -m src.train_svm
```

---

# ❌ What you should NOT do every time

You do **NOT** need to run:

✗ `pip install ...`
✗ `pip install deepface`
✗ `pip install tf-keras`
✗ `pip install opencv`

You only install ONCE — inside the venv.

---

# ✅ Why you thought you need to reinstall

Because if you **forget to activate the venv**, Windows uses the **system Python**, which has no packages.

So it looks like nothing is installed.

---

# ✅ How to check if you’re in venv

You must see this in terminal:

```
(.venv)
```

If you don’t see it — you are **not** in the environment.

---

# ✅ Make activation even easier

## Option A — Create a RUN script

### Create a file named:

```
run.ps1
```

### Put this inside:

```powershell
.\\.venv\\Scripts\\Activate.ps1
python -m src.main
```

Then just double-click or run:

```
./run.ps1
```

---

## Option B — Add shortcut command to PowerShell profile

Run:

```powershell
notepad $PROFILE
```

Add this line:

```powershell
function pro() {
    Set-Location "C:\Users\mailm\Documents\Pro"
    .\\.venv\\Scripts\\Activate.ps1
}
```

Save → reopen PowerShell → now just type:

```
pro
```

And you are:

✅ in project folder
✅ in venv

Then run:

```
python -m src.main
```

---

## Option C — Use VS Code (easiest)

1. Open folder `Pro/`
2. VS Code detects `.venv`
3. Click **Use Interpreter**
4. Now every terminal auto-activates

---

# ✅ If you want the easiest experience

Tell me which one you prefer:

### 1. Auto-activation script

### 2. One-command shortcut (`pro`)

### 3. VS Code auto environment

### 4. Batch file double-click (`run.bat`)

### 5. Desktop shortcut launcher

I will generate it **exactly for your system**.
