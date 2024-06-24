import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Hàm dự đoán giá tiếp theo
def predict_next_close(stock_data, seq_length, model, features, scaler):
    last_sequence = stock_data[features].values[-seq_length:]
    last_sequence = scaler.transform(last_sequence)
    last_sequence = np.expand_dims(last_sequence[:-1], axis=0)
    predicted_price = model.predict(last_sequence)
    predicted_price = np.concatenate([predicted_price, np.zeros((predicted_price.shape[0], len(features)-1))], axis=1)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price[0][features.index('Đóng cửa')]

st.title('Dự đoán giá cổ phiếu')

# Tải lên mô hình
model_file = st.file_uploader("lstm_model.pkl", type="pkl")
scaler_file = st.file_uploader("scaler.pkl", type="pkl")

if model_file is not None and scaler_file is not None:
    model = pickle.load(model_file)
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    
    # Tải lên dữ liệu đầu vào
    data_file = st.file_uploader("Tải lên dữ liệu cổ phiếu", type=["csv"])
    
    if data_file is not None:
        data = pd.read_csv(data_file)
        data['Ngày'] = pd.to_datetime(data['Ngày'])
        features = ['Mở cửa', 'Đóng cửa', 'Cao nhất', 'Thấp nhất', 'Trung bình', 'GD khớp lệnh KL']
        
        # Chuẩn bị dữ liệu
        data = data.sort_values(by=['Mã CK', 'Ngày'])
        data[features] = scaler.transform(data[features])
        
        # Dự đoán và tính toán lợi nhuận
        seq_length = 60
        profits = {}
        for stock in data['Mã CK'].unique():
            stock_data = data[data['Mã CK'] == stock]
            current_price = stock_data['Đóng cửa'].values[-1]
            predicted_price = predict_next_close(stock_data, seq_length, model, features, scaler)
            profit = (predicted_price - current_price) / current_price
            profits[stock] = profit
        
        # In mã cổ phiếu có lợi nhuận dự đoán cao nhất
        best_stock = max(profits, key=profits.get)
        st.write(f"Mã cổ phiếu có lợi nhuận dự đoán cao nhất là: {best_stock} với lợi nhuận dự đoán là: {profits[best_stock]*100:.2f}%")
        
        # Giả sử bạn có một số tiền để đầu tư
        investment_amount = st.number_input("Nhập số tiền đầu tư (VND):", min_value=0, value=1000000)
        potential_profit = investment_amount * profits[best_stock]
        st.write(f"Lợi nhuận dự đoán khi đầu tư vào {best_stock} là: {potential_profit:.2f} VND")
