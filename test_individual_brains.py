import requests

print("=== Testing default brain ===")
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "hunger": 90.0,
        "energy": 20.0,
        "distance_to_food": 1.0,
        "distance_to_toy": 5.0
    }
)
print(f"Default brain decision: {response.json()}")
print()

print("=== Testing individual cat brain (whiskers) ===")
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "cat_id": "whiskers",
        "hunger": 90.0,
        "energy": 20.0,
        "distance_to_food": 1.0,
        "distance_to_toy": 5.0
    }
)
print(f"Whiskers' brain decision: {response.json()}")
print()

print("=== Testing another cat (fluffy) ===")
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "cat_id": "fluffy",
        "hunger": 30.0,
        "energy": 15.0,
        "distance_to_food": 8.0,
        "distance_to_toy": 3.0
    }
)
print(f"Fluffy's brain decision: {response.json()}")
