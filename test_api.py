import requests

# Test 1: Very hungry cat near food
response = requests.post(
    "http://localhost:8000/predict",
    json={"hunger": 90.0, "energy": 20.0, "distance_to_food": 1.0, "distance_to_toy": 5.0}
)
print("Test 1 - Very hungry cat:")
print(response.json())
print()

# Test 2: Tired cat
response = requests.post(
    "http://localhost:8000/predict",
    json={"hunger": 30.0, "energy": 15.0, "distance_to_food": 8.0, "distance_to_toy": 3.0}
)
print("Test 2 - Very tired cat:")
print(response.json())
print()

# Test 3: Batch prediction
response = requests.post(
    "http://localhost:8000/predict_batch",
    json={
        "states": [
            {"hunger": 50.0, "energy": 70.0, "distance_to_food": 3.5, "distance_to_toy": 7.2},
            {"hunger": 90.0, "energy": 20.0, "distance_to_food": 1.0, "distance_to_toy": 5.0},
            {"hunger": 20.0, "energy": 10.0, "distance_to_food": 9.0, "distance_to_toy": 2.0}
        ]
    }
)
print("Test 3 - Batch prediction:")
print(response.json())
print()

# Test 4: Model info
response = requests.get("http://localhost:8000/models/latest")
print("Test 4 - Model info:")
print(response.json())
