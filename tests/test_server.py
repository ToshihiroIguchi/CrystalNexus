#!/usr/bin/env python3
"""
CIF parsing server test code
"""

import requests
import json
import time
import os

SERVER_URL = "http://localhost:5001"

def test_health_check():
    """Health check test"""
    print("=== Health Check Test ===")
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        data = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(data, indent=2)}")
        
        if response.status_code == 200 and data.get('status') == 'healthy':
            print("✅ Health check passed")
            return True
        else:
            print("❌ Health check failed")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_cif_parsing():
    """CIF parsing API test"""
    print("\n=== CIF Parsing Test ===")
    
    # Load BaTiO3.cif file
    if not os.path.exists('BaTiO3.cif'):
        print("❌ BaTiO3.cif file not found")
        return False
    
    try:
        with open('BaTiO3.cif', 'r') as f:
            cif_content = f.read()
        
        payload = {
            'cif_content': cif_content
        }
        
        print("Sending CIF content to server...")
        response = requests.post(
            f"{SERVER_URL}/parse_cif", 
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ CIF parsing successful")
            print(f"Response: {json.dumps(data, indent=2)}")
            
            # Validate important values
            expected_checks = [
                ('success', True),
                ('formula', 'BaTiO3'),
                ('space_group', str),
                ('atom_count', int),
            ]
            
            all_passed = True
            for key, expected_type in expected_checks:
                if key in data:
                    value = data[key]
                    if expected_type == str:
                        if isinstance(value, str) and value.strip():
                            print(f"✅ {key}: {value}")
                        else:
                            print(f"❌ {key}: Invalid string value")
                            all_passed = False
                    elif expected_type == int:
                        if isinstance(value, int) and value > 0:
                            print(f"✅ {key}: {value}")
                        else:
                            print(f"❌ {key}: Invalid integer value")
                            all_passed = False
                    elif isinstance(value, expected_type):
                        print(f"✅ {key}: {value}")
                    else:
                        print(f"❌ {key}: Type mismatch")
                        all_passed = False
                else:
                    print(f"❌ {key}: Missing in response")
                    all_passed = False
            
            return all_passed
            
        else:
            print(f"❌ CIF parsing failed with status {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Raw response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ CIF parsing test error: {e}")
        return False

def test_direct_cif_endpoint():
    """Direct test endpoint test"""
    print("\n=== Direct CIF Test Endpoint ===")
    try:
        response = requests.get(f"{SERVER_URL}/test_cif", timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Direct CIF test successful")
            print(f"Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ Direct CIF test failed")
            try:
                error_data = response.json()
                print(f"Error: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Raw response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Direct CIF test error: {e}")
        return False

def test_invalid_cif():
    """Invalid CIF file test"""
    print("\n=== Invalid CIF Test ===")
    
    invalid_cif = "This is not a valid CIF file"
    
    try:
        payload = {
            'cif_content': invalid_cif
        }
        
        response = requests.post(
            f"{SERVER_URL}/parse_cif", 
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code >= 400:
            print("✅ Invalid CIF properly rejected")
            try:
                error_data = response.json()
                print(f"Error response: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Raw response: {response.text}")
            return True
        else:
            print("❌ Invalid CIF was unexpectedly accepted")
            return False
            
    except Exception as e:
        print(f"❌ Invalid CIF test error: {e}")
        return False

def run_performance_test():
    """Performance test"""
    print("\n=== Performance Test ===")
    
    if not os.path.exists('BaTiO3.cif'):
        print("❌ BaTiO3.cif file not found")
        return False
    
    try:
        with open('BaTiO3.cif', 'r') as f:
            cif_content = f.read()
        
        payload = {
            'cif_content': cif_content
        }
        
        # Run 5 times and measure average time
        times = []
        for i in range(5):
            start_time = time.time()
            
            response = requests.post(
                f"{SERVER_URL}/parse_cif", 
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)
            
            if response.status_code != 200:
                print(f"❌ Request {i+1} failed")
                return False
            
            print(f"Request {i+1}: {elapsed:.3f}s")
        
        avg_time = sum(times) / len(times)
        print(f"✅ Average response time: {avg_time:.3f}s")
        
        if avg_time < 2.0:  # Pass if within 2 seconds
            print("✅ Performance test passed")
            return True
        else:
            print("❌ Performance test failed (too slow)")
            return False
            
    except Exception as e:
        print(f"❌ Performance test error: {e}")
        return False

def main():
    """Main test execution"""
    print("CIF Server API Test Suite")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("CIF Parsing", test_cif_parsing),
        ("Direct CIF Endpoint", test_direct_cif_endpoint),
        ("Invalid CIF Handling", test_invalid_cif),
        ("Performance", run_performance_test),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Results summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("💥 Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)