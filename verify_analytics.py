
import requests
import sys

def verify():
    base_url = "http://localhost:8080"
    
    # 1. Check Dashboard HTML
    try:
        resp = requests.get(f"{base_url}/analytics")
        print(f"GET /analytics: {resp.status_code}")
        if resp.status_code == 200 and "CrystalNexus Analytics" in resp.text:
            print("OK: Analytics dashboard page loaded")
        else:
            print(f"FAIL: Analytics page invalid. Status: {resp.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"FAIL: Could not connect to {base_url}/analytics: {e}")
        sys.exit(1)

    # 2. Check API
    try:
        resp = requests.get(f"{base_url}/api/analytics/summary")
        print(f"GET /api/analytics/summary: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"OK: API returned data: {data.keys()}")
        else:
            print(f"FAIL: API error. Status: {resp.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"FAIL: Could not connect to API: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify()
