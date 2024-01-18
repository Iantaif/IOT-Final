# app.py

from flask import Flask, render_template
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load("ridge_model.pkl")

# Load the latest data
df = pd.read_csv("data.csv")
df["DATE"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y")
df[["PRCP", "TMAX", "TMIN"]] = df[["PRCP", "TMAX", "TMIN"]].apply(pd.to_numeric, errors='coerce')
imputer = SimpleImputer(strategy='constant', fill_value=0)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

@app.route('/')
def index():
    # Get the last date
    last_date = df["DATE"].max()

    # Extract PRCP, TMAX, TMIN values for the last date
    last_values = df.loc[df["DATE"] == last_date, ["PRCP", "TMAX", "TMIN"]]
    # Check if data for the last date is available
    if not last_values.empty:
        last_values = last_values.iloc[0]  # Take the first row if multiple rows are found
        print(last_values)


        # Create new data for the next day
        new_date = last_date + pd.Timedelta(days=1)
        new_data = pd.DataFrame(index=[new_date], columns=["PRCP", "TMAX"])
        print(new_date)

        # Fill with the last known values
        new_data[["PRCP", "TMAX"]] = last_values[["PRCP", "TMAX"]].values
        # Ensure the index is datetime
        new_data.index = pd.to_datetime(new_data.index)

        # Predict for the next day
        prediction = model.predict(new_data)+20

        # Round and convert to int

        # Create a context for rendering the template
        context = {
            "last_date": last_date.strftime("%Y-%m-%d"),
            "prcp_last_date": last_values["PRCP"],
            "tmax_last_date": last_values["TMAX"],
            "tmin_last_date": last_values.get("TMIN", "N/A"), 
            "prediction_for_next_day": prediction
        }
        # Render the template with the context
        html_content = render_template('index.html', **context)

        # Save the rendered HTML to a file
        with open('output.html', 'w') as f:
            f.write(html_content)

        return html_content
    else:
        return "No data available for the last date."

if __name__ == '__main__':
    app.run(debug=True)
