# Retail ML & Analytics (Superstore)

A beginner-friendly **Python Machine Learning web app** built with **FastAPI** and a simple HTML page.

It uses the real **Sample - Superstore.csv** dataset to help with three retail problems.

---

## What this project does

### 1. Inventory Forecast
Estimates **next-week Quantity (units)** for stock planning.

- Choose **Overall**, a **Category**, or a **Sub-Category**
- Set a **safety buffer** (for example `0.10` = 10%)
- The app returns a forecast and a **suggested stock quantity**

This is a planning estimate only. The dataset does **not** include inventory on-hand or supplier lead times.

### 2. Discount & Profit Analysis
Estimates **Profit** for different discount levels (for example 0%, 10%, 20%, 30%, 40%, 50%).

- You choose Category, Sub-Category, Segment, Region, Quantity, and Sales
- Sales is **held constant** in the scenarios
- Results show **historical association**, not proven cause-and-effect

### 3. Customer Segmentation (RFM + K-Means)
Groups customers by:

- **Recency** – how recently they ordered
- **Frequency** – how many distinct orders they made
- **Monetary** – how much they spent (Sales)

You can:

- Load a **segment summary table**
- Optionally look up one customer by **Customer ID** to see their segment and a cautious marketing suggestion

This is **unsupervised learning**. The model groups similar customers. It does not predict a known label, and marketing suggestions are not proven campaign results.

---

## Prerequisites

- **Windows** computer
- **Python 3.10 or newer** (64-bit recommended)
  - This project’s dependencies are listed in `requirements.txt` without pinned versions
  - Python **3.10, 3.11, or 3.12** is a safe choice for most students
- **Internet** only for the first time, to download packages with `pip`
- **Command Prompt (CMD)** — these instructions use CMD, not PowerShell

### If `python` is not recognized

1. Install Python from [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. During install, tick **“Add python.exe to PATH”**
3. Close and reopen CMD
4. Check with:

```bat
python --version
```

If that fails, try:

```bat
py --version
```

If `py` works but `python` does not, replace `python` with `py` in the commands below.

---

## What is included in this project

```text
retail-ml-project/
├── main.py                 FastAPI backend (start the website from here)
├── train_models.py         Trains models from the CSV and saves files in models/
├── check_data.py           Optional: inspect the CSV
├── test_api.py             Optional: automated API checks (needs httpx)
├── requirements.txt        Python packages to install
├── README.md               This file
├── .gitignore
├── data/
│   └── Sample - Superstore.csv
├── models/                 Trained model files and metrics (already included)
└── static/
    └── index.html          Website UI
```

### Important: training vs included models

The app **needs** the files inside `models/` to start.

This shared project **already includes** trained model artifacts, so after installing packages your friend can usually start the server **without** running training first.

- **Recommended first run:** install packages → start the server
- **Run training** (`python train_models.py`) if:
  - `models/` files are missing, or
  - the app fails to load models (for example after a different Python / scikit-learn version)

Training needs the CSV in `data/Sample - Superstore.csv` (also included).

---

## Step-by-step setup (Windows CMD only)

### 1) Extract the ZIP

Extract the ZIP to any folder, for example:

`C:\Users\YourName\Desktop\retail-ml-project`

### 2) Open Command Prompt

Press **Windows key**, type **cmd**, open **Command Prompt**.

### 3) Go to the extracted project folder

Replace the path with **your** real folder path:

```bat
cd /d "PATH_TO_EXTRACTED_PROJECT"
```

Example:

```bat
cd /d "C:\Users\YourName\Desktop\retail-ml-project"
```

Check that you can see `main.py`:

```bat
dir
```

### 4) Create a virtual environment

```bat
python -m venv venv
```

### 5) Activate the virtual environment

```bat
venv\Scripts\activate.bat
```

You should see `(venv)` at the start of the line.

### 6) Upgrade pip and install dependencies

```bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 7) (Optional) Train models

Only if model files are missing or the app cannot load them:

```bat
python train_models.py
```

This can take a short while and will recreate files under `models/`.

### 8) Start the web app

```bat
python -m uvicorn main:app --reload
```

**Keep this CMD window open** while you use the website.

### 9) Open the website

In your browser, open:

- Website: http://127.0.0.1:8000/
- API docs: http://127.0.0.1:8000/docs

---

## How to stop and start again later

### Stop the server

In the CMD window where the server is running, press:

**Ctrl + C**

### Start again later

Open CMD again:

```bat
cd /d "PATH_TO_EXTRACTED_PROJECT"
venv\Scripts\activate.bat
python -m uvicorn main:app --reload
```

You usually do **not** need to reinstall packages or retrain models every time.

---

## How to try each feature

1. Open http://127.0.0.1:8000/
2. **Inventory Forecast** – pick Category / Sub-Category, set buffer, click **Run Forecast**
3. **Discount & Profit Analysis** – fill the form, click **Estimate Profit Scenarios**
4. **Customer Segmentation** – click **Load Segment Summary**, then optionally enter a **Customer ID** (sample IDs appear as suggestions)
5. Click **Load Metrics** to view saved evaluation metrics

---

## Optional: run automated tests

Only if you want extra checks. Needs one extra package:

```bat
python -m pip install httpx
python test_api.py
```

You should see `ALL_TESTS_PASSED` at the end.

---

## Main API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Website |
| GET | `/health` | Health check |
| GET | `/metrics` | Saved metrics |
| GET | `/options` | Dropdown values |
| POST | `/forecast/inventory` | Inventory forecast + suggested stock |
| POST | `/discount/scenarios` | Discount / profit scenarios |
| GET | `/segments/summary` | Segment summary |
| GET | `/segments/customer/{customer_id}` | Look up one customer |

Discount values use the same scale as the dataset: **0.2 means 20%**.

---

## Common problems

| Problem | What to try |
|---------|-------------|
| `python` is not recognized | Reinstall Python with PATH enabled, or use `py` instead of `python` |
| `activate.bat` not found | Make sure you are inside the project folder and ran `python -m venv venv` |
| Missing model files / RuntimeError | Run `python train_models.py` |
| Port 8000 already in use | Close the other program, or run: `python -m uvicorn main:app --reload --port 8001` then open http://127.0.0.1:8001/ |
| Website opens but buttons fail | Confirm the CMD server window is still running |

---

## Notes for viva / learning

- **Supervised learning:** forecast and profit models learn from known targets (Quantity / Profit)
- **Forecasting:** uses weekly history and a chronological train/test split
- **Regression:** predicts a number (units or profit)
- **Clustering:** K-Means groups customers without a target label
- **RFM:** Recency, Frequency, Monetary features built from Customer ID, Order ID, Order Date, and Sales

All outputs are **estimates for learning and planning**, not guaranteed business results.
