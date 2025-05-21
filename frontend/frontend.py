import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import os 

from data_preprocessing import preprocess
from get_data import get_binance_klines


st.set_page_config(page_title="Long Short Prediction", layout="centered")
FASTAPI_URL = os.getenv("API_URL", "https://backend:8000/predict")


def plot_candlestick(df):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Convert 'Open time' to matplotlib date format
    df['Open time'] = pd.to_datetime(df['Open time'])
    df['Open time_num'] = mdates.date2num(df['Open time'])

    width = 0.0008  

    for i in range(len(df)):
        color = 'green' if df.Close[i] >= df.Open[i] else 'red'
        # Vẽ đường high-low
        ax.plot([df['Open time_num'][i], df['Open time_num'][i]], [df.Low[i], df.High[i]], color=color)
        # Vẽ thân nến
        ax.add_patch(plt.Rectangle(
            (df['Open time_num'][i] - width/2, min(df.Open[i], df.Close[i])),
            width,
            abs(df.Close[i] - df.Open[i]),
            color=color
        ))

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    ax.set_xlabel("Thời gian")
    ax.set_ylabel("Giá")

    st.pyplot(fig)


st.title("🪙 Dự đoán xu hướng giá Crypto (BTCUSDT)")
st.markdown("Ứng dụng sử dụng mô hình học máy để dự đoán liệu giá có **tăng** hay **giảm** dựa trên các chỉ báo kỹ thuật.")

# Lấy dữ liệu
now = datetime.utcnow()
past = now - timedelta(hours=1)

df = get_binance_klines(start_time=past, end_time=now, limit=100)

st.subheader("📈 Biểu đồ giá")
plot_candlestick(df)

# Dự đoán
st.subheader("🤖 Dự đoán xu hướng tiếp theo")
if st.button("Lấy dữ liệu & Dự đoán"):
    df_feat = preprocess(df)
    if df_feat.empty:
        st.warning("Không đủ dữ liệu sau khi xử lý.")
    else:
        last_row = df_feat.iloc[-1]
        payload = last_row.to_dict()
        payload = {k: float(v) for k, v in last_row.to_dict().items()}

        try:
            res = requests.post(FASTAPI_URL, json=payload)
            if res.status_code == 200:
                pred = list(res.json())[0]
                if pred == 1:
                    st.success("📈 Mô hình dự đoán: **GIÁ TĂNG** 🚀")
                else:
                    st.error("📉 Mô hình dự đoán: **GIÁ GIẢM** ⛔")
            else:
                st.error(f"Lỗi khi gọi API: {res.status_code}")
        except Exception as e:
            st.error(f"Không thể kết nối đến FastAPI: {e}")
