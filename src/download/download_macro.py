import os
import requests
import pandas as pd


FRED_API_KEY = "1677d4a6bba57761a61c7caa5fca1b12"
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Per ora tolgo DOLLAR, perché nel tuo test DTWEXBGS ha dato 500
SERIES = {
    "VIX": "VIXCLS",
    "SP500": "SP500",
    "BRENT": "DCOILBRENTEU",
    # "DOLLAR": "DTWEXBGS",
}


def download_fred_series(series_name: str, series_id: str) -> pd.DataFrame:
    """
    Scarica una singola serie da FRED.

    Args:
        series_name: nome logico della serie nel nostro dataset finale
        series_id: codice ufficiale FRED

    Returns:
        DataFrame con colonne:
        - date
        - value
        - series
        - series_id
    """
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
    }

    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    payload = response.json()

    if "observations" not in payload:
        raise Exception(f"Risposta inattesa da FRED per {series_id}: {payload}")

    df = pd.DataFrame(payload["observations"])
    df = df[["date", "value"]].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["series"] = series_name
    df["series_id"] = series_id

    return df


def download_all_series() -> pd.DataFrame:
    """
    Scarica tutte le serie definite in SERIES.
    Se una fallisce, stampa errore e continua con le altre.
    """
    all_dfs = []
    failures = []

    for series_name, series_id in SERIES.items():
        print(f"Downloading {series_name} ({series_id})...")
        try:
            df = download_fred_series(series_name, series_id)
            all_dfs.append(df)
            print(f"  OK - {len(df)} righe scaricate")
        except Exception as e:
            print(f"  ERRORE su {series_name} ({series_id}): {e}")
            failures.append({
                "series": series_name,
                "series_id": series_id,
                "error": str(e),
            })

    if not all_dfs:
        raise Exception("Nessuna serie scaricata da FRED")

    final_df = pd.concat(all_dfs, ignore_index=True)

    return final_df, failures


if __name__ == "__main__":
    if FRED_API_KEY == "METTI_QUI_LA_TUA_FRED_API_KEY":
        raise Exception("Devi inserire la tua FRED API key nella variabile FRED_API_KEY")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "macro.csv")
    failed_file = os.path.join(output_dir, "macro_failed.csv")

    df, failures = download_all_series()
    df.to_csv(output_file, index=False)

    print(f"\n✅ Macro salvate in: {output_file}")
    print(df.head(20).to_string(index=False))

    if failures:
        failures_df = pd.DataFrame(failures)
        failures_df.to_csv(failed_file, index=False)
        print(f"\n⚠️ Alcune serie non sono state scaricate. Dettagli in: {failed_file}")