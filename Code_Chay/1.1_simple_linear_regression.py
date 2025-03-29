import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
df = pd.read_csv('D:\\Repo\\Code_ML\\Data\\price_house.csv')

# Lấy dữ liệu diện tích và giá nhà
X_train = df['area'].values
y_train = df['price'].values

# Chuẩn hóa dữ liệu
X_train = (X_train - X_train.mean()) / X_train.std()  # Chuẩn hóa X

# Thêm bias term
X_train = np.c_[np.ones(X_train.shape[0]), X_train]  # Thêm cột 1 vào X

# Khởi tạo tham số
W = np.random.randn(1)
b = np.random.randn(1)
alpha = 0.01  # Learning rate
iterations = 1000  # Số lần lặp

# Gradient Descent
m = len(y_train)
losses = []
for i in range(iterations):
    predictions = X_train[:, 1] * W + b
    errors = predictions - y_train
    dW = (1/m) * np.sum(errors * X_train[:, 1])
    db = (1/m) * np.sum(errors)
    W -= alpha * dW
    b -= alpha * db
    loss = (1/(2*m)) * np.sum(errors**2)
    losses.append(loss)

# Dự đoán
X_test = np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 100)
y_test = W * X_test + b

# Vẽ biểu đồ
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 1], y_train, label='Actual Data')
plt.plot(X_test, y_test, color='red', label='Prediction')
plt.xlabel('Normalized Area')
plt.ylabel('Price')
plt.legend()
plt.title('House Price Prediction')

plt.subplot(1, 2, 2)
plt.plot(range(iterations), losses, color='blue')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Reduction Over Time')

plt.show()


