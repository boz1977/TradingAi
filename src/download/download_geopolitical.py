import pandas as pd

def load_gpr():
    df = pd.read_csv("../../data/raw/GPR_daily.csv")
    
    df.rename(columns={
        "date": "date",
        "GPR": "gpr"
    }, inplace=True)

    return df[["date", "gpr"]]


if __name__ == "__main__":
    df = load_gpr()
    print(df.head())