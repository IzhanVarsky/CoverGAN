import os

print("Текущая директория:", os.getcwd())
for dirpath, dirnames, filenames in os.walk("../audio"):
    for filename in filenames:
        os.rename(f"../audio/{filename}", f"../audio/{filename[7:]}")
