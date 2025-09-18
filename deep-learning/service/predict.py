import matplotlib.pyplot as plt
import requests

# URL of your FastAPI endpoint
url = "http://127.0.0.1:8000/predict"

# Path to your test file (CSV or whitespace-delimited TXT)
file_path = "service/test_api_data.txt"

# Open the file in binary mode and send as multipart/form-data
with open(file_path, "rb") as f:
    response = requests.post(url, files={"file": f})

# Print the JSON response
try:
    print(response.json())

except Exception as e:
    print("Error parsing response:", e)
    print(response.text)

results = response.json()['predictions']
true_rul = response.json()['true_rul']

plt.figure(figsize=(12,5))
plt.plot(results, label="Predicted RUL")
plt.plot(true_rul, label="True RUL")
plt.xlabel("Test Sample")
plt.ylabel("RUL")
plt.grid(visible=True)
# plt.title("Predicted vs True RUL on Test Set with " + str(model.__class__.__name__) + " model")
plt.legend()
plt.show()
