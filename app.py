from flask import Flask, render_template, request, redirect, session, url_for
import pandas as pd
import numpy as np
import os
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

app = Flask(__name__)
app.secret_key = "crop_app_secret_key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DATA_FOLDER = os.path.join(BASE_DIR, "data")
CHARTS_DIR = os.path.join(BASE_DIR, "static", "charts")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# yield_df.csv columns (excluding the unnamed index):
# Area, Item, Year, hg/ha_yield, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp
NUMERIC_FEATURES = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
TARGET = "hg/ha_yield"

# In-memory state
state = {
    "dataset_path": None,
    "dataframe": None,
    "model": None,
    "r2": None,
    "mae": None,
    "rmse": None,
    "feature_importances": None,
    "le_area": None,
    "le_item": None,
    "users": {"admin": "admin"},
}

# Average local weather stats per city (unchanged)
CITY_WEATHER_AVG = {
    "chennai":    {"temp": 30, "humidity": 75, "rain_chance": 0.5},
    "hyderabad":  {"temp": 28, "humidity": 55, "rain_chance": 0.3},
    "bengaluru":  {"temp": 24, "humidity": 60, "rain_chance": 0.35},
    "mumbai":     {"temp": 29, "humidity": 80, "rain_chance": 0.55},
    "delhi":      {"temp": 27, "humidity": 45, "rain_chance": 0.2},
    "kolkata":    {"temp": 29, "humidity": 78, "rain_chance": 0.5},
    "pune":       {"temp": 26, "humidity": 55, "rain_chance": 0.3},
    "jaipur":     {"temp": 28, "humidity": 40, "rain_chance": 0.15},
    "lucknow":    {"temp": 27, "humidity": 50, "rain_chance": 0.25},
    "amritsar":   {"temp": 25, "humidity": 50, "rain_chance": 0.25},
}


def login_required(func):
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper


def load_default_dataset():
    """Load yield_df.csv as the primary dataset."""
    path = os.path.join(DATA_FOLDER, "yield_df.csv")
    state["dataset_path"] = path
    df = pd.read_csv(path)
    # Drop the unnamed index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    state["dataframe"] = df


def get_greeting():
    """Return a time-based greeting string."""
    from datetime import datetime
    hour = datetime.now().hour
    if hour < 12:
        return "morning"
    elif hour < 17:
        return "afternoon"
    else:
        return "evening"


# ── Auth routes ──────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        users = state["users"]
        if users.get(username) == password:
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid username or password")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            return render_template("signup.html", error="Username and password are required.")
        if username in state["users"]:
            return render_template("signup.html", error="That username already exists.")

        state["users"][username] = password
        session["logged_in"] = True
        session["username"] = username
        return redirect(url_for("dashboard"))
    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ── Dashboard ────────────────────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    df = state["dataframe"]
    total_records = "{:,}".format(len(df)) if df is not None else "—"
    total_countries = df["Area"].nunique() if df is not None else "—"
    total_crops = df["Item"].nunique() if df is not None else "—"
    model_accuracy = "—"
    if state["r2"] is not None:
        model_accuracy = "{:.1f}%".format(state["r2"] * 100)
    return render_template(
        "dashboard.html",
        greeting=get_greeting(),
        total_records=total_records,
        total_countries=total_countries,
        total_crops=total_crops,
        model_accuracy=model_accuracy,
    )


# ── Dataset routes ───────────────────────────────────────────────────────────

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    message = None
    if request.method == "POST":
        file = request.files.get("dataset")
        if file and file.filename.endswith(".csv"):
            save_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(save_path)
            state["dataset_path"] = save_path
            df = pd.read_csv(save_path)
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])
            state["dataframe"] = df
            return redirect(url_for("preview"))
        message = "Please choose a valid .csv file."
    return render_template("upload.html", message=message)


@app.route("/use_sample")
@login_required
def use_sample():
    load_default_dataset()
    return redirect(url_for("preview"))


@app.route("/preview")
@login_required
def preview():
    if state["dataframe"] is None:
        return redirect(url_for("upload"))
    df = state["dataframe"]
    columns = list(df.columns)
    # Determine which column index holds "Item" (1-based for template)
    item_col_idx = columns.index("Item") + 1 if "Item" in columns else 0
    crop_list = sorted(df["Item"].unique().tolist()) if "Item" in columns else []

    # Send ALL rows to template; client-side JS handles pagination
    table_data = df.values.tolist()

    return render_template(
        "preview.html",
        rows=len(df),
        columns=columns,
        table_data=table_data,
        item_col_idx=item_col_idx,
        crop_list=crop_list,
        num_features=len(columns) - 1,   # minus target or index
        num_crops=len(crop_list),
        num_countries=df["Area"].nunique() if "Area" in columns else 0,
    )


# ── Model training ───────────────────────────────────────────────────────────

@app.route("/train")
@login_required
def train():
    if state["dataframe"] is None:
        return redirect(url_for("upload"))
    df = state["dataframe"].copy()

    # Encode categorical features
    le_area = LabelEncoder()
    le_item = LabelEncoder()
    df["Area_enc"] = le_area.fit_transform(df["Area"])
    df["Item_enc"] = le_item.fit_transform(df["Item"])

    feature_cols = ["Area_enc", "Item_enc", "Year"] + NUMERIC_FEATURES
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    # Feature importances
    importances = model.feature_importances_
    feat_names = ["Country", "Crop", "Year", "Rainfall", "Pesticides", "Temperature"]
    fi_list = sorted(
        [{"name": n, "value": v} for n, v in zip(feat_names, importances)],
        key=lambda x: x["value"],
        reverse=True,
    )

    state["model"] = model
    state["r2"] = r2
    state["mae"] = mae
    state["rmse"] = rmse
    state["feature_importances"] = fi_list
    state["le_area"] = le_area
    state["le_item"] = le_item

    # Generate charts
    make_distribution_chart(df)
    make_feature_importance_chart(feat_names, importances)
    make_yield_by_crop_chart(df)

    dataset_name = os.path.basename(state["dataset_path"]) if state["dataset_path"] else "Unknown"

    return render_template(
        "train_result.html",
        dataset_name=dataset_name,
        train_samples=len(X_train),
        test_samples=len(X_test),
        num_features=len(feature_cols),
        r2_score=round(r2 * 100, 1),
        mae="{:,.0f}".format(mae),
        rmse="{:,.0f}".format(rmse),
        feature_importances=fi_list,
    )


def make_distribution_chart(df):
    """Bar chart of record count per crop type."""
    counts = df["Item"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(counts.index, counts.values, color="#245C36", edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Number of Records", fontsize=11)
    ax.set_title("Crop Distribution in Dataset", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "distribution.png"), dpi=120)
    plt.close()


def make_feature_importance_chart(names, importances):
    """Horizontal bar chart of feature importances."""
    sorted_idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([names[i] for i in sorted_idx], importances[sorted_idx], color="#D6A84F", edgecolor="white")
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title("Feature Importance (Random Forest)", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "feature_importance.png"), dpi=120)
    plt.close()


def make_yield_by_crop_chart(df):
    """Bar chart of average yield per crop."""
    avg = df.groupby("Item")[TARGET].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(avg.index, avg.values, color="#163D28", edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Average Yield (hg/ha)", fontsize=11)
    ax.set_title("Average Yield by Crop Type", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "yield_by_crop.png"), dpi=120)
    plt.close()


# ── Yield prediction ────────────────────────────────────────────────────────

@app.route("/predict_yield", methods=["GET", "POST"])
@login_required
def predict_yield_page():
    df = state["dataframe"]
    areas = sorted(df["Area"].unique().tolist()) if df is not None else []
    items = sorted(df["Item"].unique().tolist()) if df is not None else []

    result = None
    error = None
    selected_area = None
    selected_item = None
    selected_year = None
    selected_rain = None
    selected_pest = None
    selected_temp = None

    if request.method == "POST":
        if state["model"] is None:
            error = "Train the model first from the Train Model page."
        else:
            try:
                area_val = request.form["area"]
                item_val = request.form["item"]
                year_val = int(request.form["year"])
                rain_val = float(request.form["rainfall"])
                pest_val = float(request.form["pesticides"])
                temp_val = float(request.form["avg_temp"])

                selected_area = area_val
                selected_item = item_val
                selected_year = year_val
                selected_rain = rain_val
                selected_pest = pest_val
                selected_temp = temp_val

                # Encode area and item
                le_area = state["le_area"]
                le_item = state["le_item"]

                if area_val not in le_area.classes_:
                    error = f"Country '{area_val}' was not in the training data."
                elif item_val not in le_item.classes_:
                    error = f"Crop '{item_val}' was not in the training data."
                else:
                    area_enc = le_area.transform([area_val])[0]
                    item_enc = le_item.transform([item_val])[0]

                    features = [[area_enc, item_enc, year_val, rain_val, pest_val, temp_val]]
                    result = state["model"].predict(features)[0]

            except (KeyError, ValueError) as e:
                error = f"Invalid input: {e}"

    return render_template(
        "predict_yield.html",
        areas=areas,
        items=items,
        result=result,
        error=error,
        selected_area=selected_area,
        selected_item=selected_item,
        selected_year=selected_year,
        selected_rain=selected_rain,
        selected_pest=selected_pest,
        selected_temp=selected_temp,
    )


# ── Weather ──────────────────────────────────────────────────────────────────

@app.route("/weather", methods=["GET", "POST"])
@login_required
def weather():
    forecast = None
    recommendation = None
    city_used = None
    if request.method == "POST":
        city = request.form.get("city", "").strip().lower()
        if city in CITY_WEATHER_AVG:
            avg = CITY_WEATHER_AVG[city]
            forecast = []
            rainy_days = 0
            for day in range(1, 4):
                will_rain = random.random() < avg["rain_chance"]
                if will_rain:
                    rainy_days += 1
                forecast.append({
                    "day": day,
                    "temp": round(avg["temp"] + random.uniform(-2, 2), 1),
                    "humidity": round(avg["humidity"] + random.uniform(-5, 5), 1),
                    "condition": "Rain" if will_rain else "Clear",
                })
            recommendation = (
                "Heavy rain expected in the coming days — hold off on fertilizer."
                if rainy_days >= 2
                else "No major rain expected — safe to add fertilizer."
            )
            city_used = city
    return render_template(
        "weather.html",
        forecast=forecast,
        recommendation=recommendation,
        city=city_used,
        cities=sorted(CITY_WEATHER_AVG.keys()),
    )


# ── Analytics ────────────────────────────────────────────────────────────────

@app.route("/analytics")
@login_required
def analytics():
    chart_exists = os.path.exists(os.path.join(CHARTS_DIR, "distribution.png"))
    fi_chart_exists = os.path.exists(os.path.join(CHARTS_DIR, "feature_importance.png"))
    yield_chart_exists = os.path.exists(os.path.join(CHARTS_DIR, "yield_by_crop.png"))
    return render_template(
        "analytics.html",
        chart_exists=chart_exists,
        fi_chart_exists=fi_chart_exists,
        yield_chart_exists=yield_chart_exists,
    )


# ── Start ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_default_dataset()
    app.run(debug=True)
