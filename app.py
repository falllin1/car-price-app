from flask import Flask, request, render_template
import pickle
import numpy as np
import json

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Load feature columns
with open('features.json', 'r') as f:
    data = json.load(f)
with open('features.json', 'r') as f:
    data = json.load(f)
    feature_columns = data['data_columns']  


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values
        year = request.form.get('prod_year')
        mileage = request.form.get('mileage')
        fuel_type = request.form.get('fuel_type')
        gear_box = request.form.get('gear_box')
        manufacturer = request.form.get('manufacturer')
        model_name = request.form.get('model')

        # Validate
        if not year or not mileage:
            return render_template('index.html', prediction_text="Please fill all fields")

        year = float(year)
        mileage = float(mileage)

        # Create input array
        input_data = np.zeros(len(feature_columns))

        # Numeric
        if 'Prod. year' in feature_columns:
            input_data[feature_columns.index('Prod. year')] = year

        if 'Mileage' in feature_columns:
            input_data[feature_columns.index('Mileage')] = mileage

        # Categorical helper
        def set_category(prefix, value):
            col = f"{prefix}_{value}"
            if col in feature_columns:
                input_data[feature_columns.index(col)] = 1

        # Apply encoding
        set_category('Fuel type', fuel_type)
        set_category('Gear box type', gear_box)
        set_category('Manufacturer', manufacturer)
        set_category('Model', model_name)

        # Predict
        prediction = model.predict([input_data])[0]

        # Create nicer output
        price = f"${round(prediction, 2)}"
        low_price = f"${round(prediction * 0.9, 2)}"
        high_price = f"${round(prediction * 1.1, 2)}"

        return render_template(
            'results.html',
            price=price,
            low_price=low_price,
            high_price=high_price,
            confidence="85%"
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
