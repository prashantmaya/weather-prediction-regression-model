# Test 1: Health Check
curl http://localhost:8001/health

# Expected response:
# {"status":"healthy","version":"1.0.0","model_loaded":true,"timestamp":"2025-10-04T..."}

# Test 2: Root Endpoint
curl http://localhost:8001/

# Test 3: Make a Prediction
curl -X POST "http://localhost:8001/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "Location": "Sydney",
           "MinTemp": 18.0,
           "MaxTemp": 25.0,
           "Rainfall": 0.0,
           "Evaporation": 4.0,
           "Sunshine": 8.0,
           "WindGustDir": "NE",
           "WindGustSpeed": 35.0,
           "WindDir9am": "N",
           "WindDir3pm": "NE",
           "WindSpeed9am": 15.0,
           "WindSpeed3pm": 20.0,
           "Humidity9am": 70.0,
           "Humidity3pm": 60.0,
           "Pressure9am": 1015.0,
           "Pressure3pm": 1013.0,
           "Cloud9am": 5.0,
           "Cloud3pm": 4.0,
           "Temp9am": 20.0,
           "Temp3pm": 24.0,
           "RainToday": "No"
         }'

# Expected response:
# {"will_rain_tomorrow":false,"probability":0.23,"timestamp":"2025-10-04T..."}

# Test 4: Get Model Info
curl http://localhost:8001/api/v1/model/info