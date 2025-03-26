import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
df = pd.read_csv('D:\\Repo\\Code_ML\\Data\\price_house.csv')
print(f"Số lượng dòng trong dataset: {len(df)}")

# Kiểm tra thông tin cơ bản của dataset
df.info()
print(df.describe())

# Lấy dữ liệu diện tích (area) làm biến đầu vào (X) và giá nhà (price) làm biến mục tiêu (y)
X = df['area'].values
y = df['price'].values

# Chia dữ liệu thành tập train (80%) và tập test (20%)
def train_test_split_manual(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2, random_state=42)

# Tính toán hệ số W và b theo công thức hồi quy tuyến tính đơn giản
mean_x, mean_y = np.mean(X_train), np.mean(y_train)
W = np.sum((X_train - mean_x) * (y_train - mean_y)) / np.sum((X_train - mean_x) ** 2)
b = mean_y - W * mean_x

print(f"Hệ số W (slope): {W:.2f}")
print(f"Hệ số b (intercept): {b:.2f}")

# Dự đoán giá nhà trên tập test
y_pred = W * X_test + b

# Đánh giá mô hình
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Trực quan hóa kết quả
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')  # Điểm dữ liệu thực tế
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Line')  # Đường hồi quy
plt.xlabel('Diện tích (m²)')
plt.ylabel('Giá nhà (triệu đồng)')
plt.legend()
plt.title('Hồi quy tuyến tính: Dự đoán giá nhà theo diện tích')
plt.show()

