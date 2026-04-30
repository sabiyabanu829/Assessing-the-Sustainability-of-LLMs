# 🛠️ Environment Setup & Installation Guide

This guide explains how to **install Python**, **create a virtual environment**, **install required dependencies**, and **run the evaluation scripts**.

---

## 🐍 Python Version

All experiments were conducted using:

```
Python 3.12.3
```

⚠️ Using a different Python version may lead to incompatibilities with some packages.

---

## 📦 Step 1: Install Python 3.12.3

Download Python from the official website:

👉 https://www.python.org/downloads/release/python-3123/

During installation:
- ✅ Check **“Add Python to PATH”**
- ✅ Install `pip`

Verify installation:

```bash
python --version
```

Expected output:
```text
Python 3.12.3
```

---

## 🧪 Step 2: Create a Virtual Environment

From the project root directory:

### Windows (PowerShell / CMD)
```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

Once activated, your terminal should show:
```text
(venv)
```

---

## 📥 Step 3: Install Requirements

Ensure a file named `requirements.txt` exists in the project root.


Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Step 4: Run the Scripts

### Run Answer Generation Pipeline
```bash
python AnswerGeneration-Code.py
```

### Run Energy Evaluation Pipeline
```bash
python Energy-Evaluation-Pipeline.py
```

### Run LLM-as-a-Judge Evaluation
```bash
python CodeJudge.py
```

This setup ensures **reproducibility**, **dependency isolation**, and **consistent results**.
