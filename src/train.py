from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd 
import os 
import joblib


def train():
    df = pd.read_csv("data/Iris.csv")
    x = df.drop(columns=['Id','Species'])
    y = df['Species']

    xtr,xte,ytr,yte = train_test_split(x,y,random_state=42)
    model = RandomForestClassifier()
    model.fit(xtr,ytr)
    y_pred = model.predict(xte)

    acs = accuracy_score(yte,y_pred)

    print(f"Acuracy - {acs}")

    if acs < 0.7:
        raise ValueError("Model is not accurate")

    os.makedirs('model',exist_ok=True)
    joblib.dump(model,"model/model.pkl")

    print("Model saved")


if __name__ == "__main__":
    train()    