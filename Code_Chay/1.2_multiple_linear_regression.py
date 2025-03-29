import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu
file_path = "D:\\Repo\\Code_ML\\Data\\price_house.csv"  # Thay thế bằng đường dẫn thực tế
df = pd.read_csv(file_path)

# Chuyển đổi biến phân loại thành số
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes

# Chia dữ liệu thành features (X) và target (y)
X = df.drop(columns=['price']).values  # Loại bỏ cột giá nhà
y = df['price'].values.reshape(-1, 1)

# Chuẩn hóa dữ liệu
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Thêm cột bias (1s) vào X để tính toán hệ số hồi quy
X = np.c_[np.ones(X.shape[0]), X]

# Chia dữ liệu thành tập train và test
train_size = int(0.8 * X.shape[0])
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Tính toán hệ số hồi quy bằng công thức bình phương tối thiểu thông thường
beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Dự đoán trên tập test
y_pred = X_test @ beta

# Đánh giá mô hình
mae = np.mean(np.abs(y_test - y_pred))
mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)
total_variance = np.sum((y_test - np.mean(y_test)) ** 2)
explained_variance = np.sum((y_pred - np.mean(y_test)) ** 2)
r2 = explained_variance / total_variance

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2 Score: {r2:.4f}")

# Vẽ biểu đồ thực tế vs dự đoán
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test.flatten(), y=y_pred.flatten(), alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', linewidth=2)
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Biểu đồ Giá thực tế vs Giá dự đoán")
plt.show()
