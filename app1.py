import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import pickle
from tensorflow.keras.models import load_model

# Load mô hình và scaler
model = load_model('my_model.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Đọc dữ liệu từ file CSV
data = pd.read_csv('_Dữ_liệu_giao_dịch_ngày__202406152152.csv')

# Chọn các đặc trưng quan trọng
features = ['Mở cửa', 'Đóng cửa', 'Cao nhất', 'Thấp nhất', 'Trung bình', 'GD khớp lệnh KL']
data = data[['Ngày', 'Mã CK'] + features]
data = data.sort_values(by=['Mã CK', 'Ngày'])
data[features] = scaler.transform(data[features])

# Tạo chuỗi thời gian cho LSTM
def create_sequences(data, seq_length, features):
    sequences = []
    for stock in data['Mã CK'].unique():
        stock_data = data[data['Mã CK'] == stock][features].values
        for i in range(len(stock_data) - seq_length):
            sequences.append(stock_data[i:i+seq_length])
    return np.array(sequences)

seq_length = 60

# Dự đoán giá đóng cửa ngày tiếp theo
def predict_next_close(stock_data, seq_length, model, features, scaler):
    last_sequence = stock_data[features].values[-seq_length:]
    last_sequence = scaler.transform(last_sequence)
    last_sequence = np.expand_dims(last_sequence, axis=0)
    predicted_price = model.predict(last_sequence)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price[0][features.index('Đóng cửa')]

# Tính toán lợi nhuận dự kiến cho mỗi mã chứng khoán
profits = {}
for stock in data['Mã CK'].unique():
    stock_data = data[data['Mã CK'] == stock]
    current_price = stock_data['Đóng cửa'].values[-1]
    predicted_price = predict_next_close(stock_data, seq_length, model, features, scaler)
    profit = (predicted_price - current_price) / current_price
    profits[stock] = profit

# Chọn mã chứng khoán có lợi nhuận dự kiến cao nhất
top_stocks = sorted(profits, key=profits.get, reverse=True)[:10]

# Triển khai trên Streamlit
st.title("Stock Prediction System")
budget = st.number_input("Enter your budget:", min_value=0, step=1000)

if st.button("Get Recommendations"):
    # Hiển thị 10 mã chứng khoán có lợi nhuận cao nhất
    st.write("Top 10 recommended stocks:")
    for stock in top_stocks:
        st.write(f"Stock: {stock}, Predicted Profit: {profits[stock]:.2%}")
