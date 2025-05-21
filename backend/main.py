from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
import pickle
import pandas as pd 
from datetime import datetime
import numpy as np 

from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse


class InputItem(BaseModel):
    Low: float
    High: float
    Close: float
    SMA_10: float
    RSI_14: float
    ATRr_14: float
    ADX_14: float
    DMP_14: float
    DMN_14: float
    SKEW_30: float
    SLOPE_1: float
    BBL_5_2_0: float = Field(alias="BBL_5_2.0")
    BBU_5_2_0: float = Field(alias="BBU_5_2.0")
    MACD_12_26_9: float
    MACDs_12_26_9: float


app = FastAPI() 
templates = Jinja2Templates(directory="template")

model_dir = "/model/model.pkl"
with open (model_dir, mode='rb' ) as file: 
    model = pickle.load(file)


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(data: InputItem):
    arr = np.array([
        [
            data.Low,
            data.High,
            data.Close,
            data.SMA_10, 
            data.RSI_14, 
            data.ATRr_14, 
            data.ADX_14, 
            data.DMP_14, 
            data.DMN_14, 
            data.SKEW_30, 
            data.SLOPE_1, 
            data.BBL_5_2_0, 
            data.BBU_5_2_0, 
            data.MACD_12_26_9, 
            data.MACDs_12_26_9
        ]
    ])

    pred = model.predict(arr)[0]
    return {int(pred)}

