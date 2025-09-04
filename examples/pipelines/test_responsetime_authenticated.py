import os
import requests
import json

max_response_time = 0.5

def send_request(endpoint, headers=None, payload=None):
    if payload:
        response = requests.post(endpoint, headers=headers, json=payload)
    else:
        response = requests.get(endpoint, headers=headers)
    return response

def test_responsetime(endpoint, headers=None, payload=None):
    response = send_request(endpoint, headers, payload)

    if response.status_code == 200:
        response_time = response.elapsed.total_seconds()
    else:
        raise Exception(f"Response status code is {response.status_code}. Response: {response.text}")

    if response_time > max_response_time:
        raise Exception(f"Response took {response_time} which is greater than {max_response_time}")

    print(f"Response time was OK at {response_time} seconds")

    with open("responsetime_result.json", "w") as f:
        json.dump({
            "response_time": response_time,
            "status_code": response.status_code
        }, f)

if __name__ == '__main__':
    # Option 1: Test health endpoint (no auth required)
    health_endpoint = "https://granite-8b-code-instruct-maas-apicast-production.apps.llmaas.llmaas.redhatworkshops.io:443/health"
    
    # Option 2: Test completions endpoint (requires auth)
    api_endpoint = "https://granite-8b-code-instruct-maas-apicast-production.apps.llmaas.llmaas.redhatworkshops.io:443/v1/completions"
    
    # API Token from the codebase
    API_TOKEN = "65fc80b0d55be557b1365687ddb771d6"
    
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
        "accept": "application/json"
    }
    
    payload = {
        "model": "granite-8b-code-instruct-128k",
        "prompt": "Hello",
        "max_tokens": 10,
        "temperature": 0.01
    }
    
    try:
        print("Testing health endpoint...")
        test_responsetime(health_endpoint)
        print("Health endpoint test passed!")
    except Exception as e:
        print(f"Health endpoint test failed: {e}")
    
    try:
        print("\nTesting API endpoint...")
        test_responsetime(api_endpoint, headers, payload)
        print("API endpoint test passed!")
    except Exception as e:
        print(f"API endpoint test failed: {e}") 