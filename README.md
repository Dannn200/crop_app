# Crop Advisor

A simple Flask app for crop recommendation and yield estimation using
Naive Bayes — no external APIs used anywhere (weather is simulated from
fixed city averages, yield is a fixed average-per-acre lookup).

## Setup

```
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

Log in with `admin` / `admin`.

## Flow

1. **Upload** — upload your own CSV (columns: N, P, K, temperature,
   humidity, ph, rainfall, label) or click "Use the built-in sample
   dataset" to use the one included in `data/crop_recommendation.csv`.
2. **Preview** — see the loaded data as a table.
3. **Train** — trains a GaussianNB model and reports test accuracy.
4. **Predict Crop** — enter soil/climate readings, get a recommended crop.
5. **Predict Yield** — pick a crop and land area, get an estimated yield
   (from a fixed kg-per-acre table).
6. **Weather** — pick a city, get a simulated 3-day outlook and a
   fertilizer recommendation based on it.
7. **Chart** — bar chart of how many records per crop are in the
   training data.

## Notes

- The included `data/crop_recommendation.csv` is a synthetic dataset
  (660 rows, 22 crops) generated to match the structure of the well-known
  Kaggle "Crop Recommendation Dataset" (N, P, K, temperature, humidity,
  ph, rainfall → label). Swap in the real Kaggle CSV for more realistic
  numbers if you want.
- Login is hardcoded (no database), matching a simple demo-style app.
- All app state is kept in memory, so it resets each time you restart
  the server.
