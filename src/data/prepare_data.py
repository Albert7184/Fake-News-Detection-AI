import pandas as pd

# 1 Load dataset
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

print("Fake shape:", fake.shape)
print("Real shape:", real.shape)

# 2 Thêm label
fake["label"] = 0
real["label"] = 1

# 3 Chỉ lấy cột text
fake = fake[["text", "label"]]
real = real[["text", "label"]]

# 4 Gộp dataset
data = pd.concat([fake, real])

# 5 Shuffle dữ liệu
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("Total dataset:", data.shape)

# 6 Lưu dataset mới
data.to_csv("data/news.csv", index=False)

print("Dataset saved to data/news.csv")