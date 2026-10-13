"""Illustrative agribusiness market-price analysis workflow.

Input: data/market_prices.csv with `date` and `modal_price` columns.
Update column names and units after inspecting your chosen public dataset.
"""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Explore monthly agricultural market prices.")
    parser.add_argument("--input", default="data/market_prices.csv", help="CSV input path")
    parser.add_argument("--output-dir", default="outputs", help="Directory for output charts")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find {input_path}. Download a permitted public dataset, save it there, "
            "and ensure it has `date` and `modal_price` columns (or update this script)."
        )

    df = pd.read_csv(input_path)
    required = {"date", "modal_price"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}. Update the script to match your dataset."
        )

    print("Rows and columns:", df.shape)
    print("\nMissing values by column:\n", df.isna().sum())

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["modal_price"] = pd.to_numeric(df["modal_price"], errors="coerce")
    clean = df.dropna(subset=["date", "modal_price"]).copy()

    if clean.empty:
        raise ValueError("No usable rows remain after parsing date and modal_price.")

    monthly = clean.set_index("date")["modal_price"].resample("MS").mean().dropna()
    if monthly.empty:
        raise ValueError("No monthly price series could be created from the input data.")

    print("\nMonthly price summary:\n", monthly.describe())

    ax = monthly.plot(title="Monthly Average Agricultural Market Price")
    ax.set_xlabel("Month")
    ax.set_ylabel("Price (source units)")
    plt.tight_layout()
    chart_path = output_dir / "monthly_price_trend.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Saved chart: {chart_path}")

    comparison = pd.DataFrame({
        "actual": monthly,
        "baseline_previous_period": monthly.shift(1)
    }).dropna()
    if not comparison.empty:
        mae = (comparison["actual"] - comparison["baseline_previous_period"]).abs().mean()
        print(f"Previous-period baseline MAE: {mae:.2f}")
    else:
        print("Not enough monthly observations to calculate baseline MAE.")


if __name__ == "__main__":
    main()
