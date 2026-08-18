from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv(Path(__file__).resolve().parent / "Ecommerce Customers.csv")

# Prepare features and target
X = df[["Avg. Session Length","Time on App", "Time on Website","Length of Membership"]]
y = df["Yearly Amount Spent"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.3, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Start user interaction
print("Welcome to the E-Commerce Yearly Amount Predictor!")
while True:
    print("\nPlease enter customer details:")
    try:
        avg_session = float(input("Avg. Session Length: "))
        time_app = float(input("Time on App: "))
        time_website = float(input("Time on Website: "))
        membership_length = float(input("Length of Membership: "))
    except ValueError:
        print("Please enter a valid number.")
        continue

    # Build input DataFrame
    new_data = pd.DataFrame({
        "Avg. Session Length": [avg_session],
        "Time on App": [time_app],
        "Time on Website": [time_website],
        "Length of Membership": [membership_length]
    })

    # Predict
    prediction = model.predict(new_data)
    print(f"\nPredicted Yearly Amount Spent: ${prediction[0]:.2f}")

    # Ask if user wants to predict another
    cont = input("\nDo you want to predict for another customer? (yes/no): ").lower()
    if cont != "yes":
        print("Thank you for using the predictor! Goodbye.")
        break