import os
import pandas as pd


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(__file__))

    prices_file = os.path.join(base_dir, "data", "raw", "prices.csv")
    macro_file = os.path.join(base_dir, "data", "raw", "macro.csv")
    output_dir = os.path.join(base_dir, "data", "processed")
    output_file = os.path.join(output_dir, "dataset.csv")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Lettura prezzi: {prices_file}")
    prices = pd.read_csv(prices_file)
    prices["Date"] = pd.to_datetime(prices["Date"])

    df = prices.copy()

    if os.path.exists(macro_file):
        print(f"Lettura macro: {macro_file}")
        macro = pd.read_csv(macro_file)
        macro["date"] = pd.to_datetime(macro["date"])

        # Trasformo le righe in colonne:
        # date | series | value  -->  date | VIX | SP500 | BRENT | DOLLAR
        macro_pivot = macro.pivot(index="date", columns="series", values="value")

        # Merge sui prezzi
        df = df.merge(macro_pivot, left_on="Date", right_index=True, how="left")

        # Forward fill:
        # se in un certo giorno non c'è valore macro, uso l'ultimo disponibile
        for col in ["VIX", "SP500", "BRENT", "DOLLAR"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].ffill()
    else:
        print("macro.csv non trovato: merge macro saltato")

    df.to_csv(output_file, index=False)

    print(f"\n✅ Dataset creato in: {output_file}")
    print(df.head(10).to_string(index=False))