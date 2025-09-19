import matplotlib.pyplot as plt
import requests
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

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

# # Create the figure and axis
# fig, ax = plt.subplots(figsize=(14,5))
# x = list(range(len(results)))
# y_pred = results
# y_hat = true_rul
# line1, = ax.plot([], [], marker='o', color='royalblue', label='Estimated RUL')
# line2, = ax.plot([], [], color='steelblue', label='Actual RUL')

# ax.set_xlim(0, len(y_pred) - 1)
# ax.set_ylim(0, max(len(y_pred), len(y_hat)))
# ax.legend()
# ax.grid(visible=True)
# ax.set_xlabel("Operation Cycle")
# ax.set_ylabel("RUL")
# ax.set_title("Estimated vs True RUL on Test Engine with LSTM model")


# Create the figure and axis
figure = plt.figure(figsize=(14,5))
x = list(range(len(results)))
y_pred = results
y_hat = true_rul
line1, = plt.plot([], [], marker='o', color='royalblue', label='Estimated RUL')
line2, = plt.plot([], [], color='steelblue', label='True RUL')

plt.xlim(0, len(y_pred) - 1)
plt.ylim(0, max(len(y_pred), len(y_hat)))
plt.legend()
plt.grid(visible=True)
plt.xlabel("Operation Cycle")
plt.ylabel("RUL")
plt.title("Estimated vs True RUL on Test Engine with LSTM model")

# Update function for the animation
def update(frame):
    line1.set_data(x[:frame], y_pred[:frame])
    line2.set_data(x[:frame], y_hat[:frame])
    return line1, line2

# Create the animation
ani = FuncAnimation(figure, update, frames=len(x) + 1, interval=50, blit=True)

# Save as GIF
writer = PillowWriter(fps=20)  # fps = frames per second
ani.save("service/lstm.gif", writer=writer)

# Show the animation
plt.show()
