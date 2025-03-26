import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Đọc dữ liệu từ file CSV
df = pd.read_csv('D:\\Repo\\Code_ML\\Data\\price_house.csv')
print(f"Số lượng dòng trong dataset: {len(df)}")

# Kiểm tra thông tin cơ bản của dataset
df.info()
print(df.describe())

# Lấy dữ liệu diện tích (area) làm biến đầu vào (X) và giá nhà (price) làm biến mục tiêu (y)
X = df[['area']].values  # Định dạng thành ma trận 2D để phù hợp với sklearn
y = df['price'].values

# Chia dữ liệu thành tập train (80%) và tập test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình hồi quy tuyến tính
model = LinearRegression()

# Huấn luyện mô hình với tập dữ liệu train
model.fit(X_train, y_train)

# Dự đoán giá nhà trên tập test
y_pred = model.predict(X_test)

# Đánh giá mô hình
mae = mean_absolute_error(y_test, y_pred)  # MAE: Sai số tuyệt đối trung bình
mse = mean_squared_error(y_test, y_pred)  # MSE: Sai số bình phương trung bình
rmse = np.sqrt(mse)  # RMSE: Căn bậc hai của MSE
r2 = r2_score(y_test, y_pred)  # R^2: Hệ số xác định

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

# Hiển thị hệ số W (slope) và b (intercept) của mô hình
print(f"Hệ số W (slope): {model.coef_[0]:.2f}")
print(f"Hệ số b (intercept): {model.intercept_:.2f}")
