#!/usr/bin/env python3
"""
Enhanced Endpoint Connectivity Test
Tests all server endpoints and detects API mismatches
"""

import requests
import json
import time
import os
from urllib.parse import urlparse

def test_all_endpoints():
    """Test all available endpoints"""
    
    server_host = os.getenv('SERVER_HOST', '127.0.0.1')
    server_port = int(os.getenv('SERVER_PORT', 8080))
    base_url = f"http://{server_host}:{server_port}"
    
    print("=== Endpoint Connectivity Test ===")
    print(f"Testing server: {base_url}")
    
    # Expected endpoints based on server implementation
    endpoints = {
        'GET': [
            '/',
            '/health',
            '/test_cif',
            '/cache_info',
            '/sample_files',
            '/sample_cif/BaTiO3.cif'
        ],
        'POST': [
            '/parse_cif',
            '/create_supercell',
            '/delete_atoms',
            '/replace_atoms'
        ]
    }
    
    results = {}
    
    # Test GET endpoints
    for endpoint in endpoints['GET']:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            results[f"GET {endpoint}"] = {
                'status_code': response.status_code,
                'success': response.status_code in [200, 201],
                'content_type': response.headers.get('content-type', ''),
                'size': len(response.text)
            }
            status = "✅" if results[f"GET {endpoint}"]['success'] else "❌"
            print(f"  {status} GET {endpoint}: {response.status_code}")
            
        except Exception as e:
            results[f"GET {endpoint}"] = {'error': str(e), 'success': False}
            print(f"  ❌ GET {endpoint}: {e}")
    
    # Test POST endpoints with sample data
    sample_cif = """data_test
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_space_group_name_H-M_alt 'P1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 0.0 0.0 0.0
"""
    
    post_data = {
        '/parse_cif': {'cif_content': sample_cif},
        '/create_supercell': {
            'cif_content': sample_cif,
            'a_multiplier': 2,
            'b_multiplier': 2,
            'c_multiplier': 2
        },
        '/delete_atoms': {
            'cif_content': sample_cif,
            'atom_indices': [0]
        },
        '/replace_atoms': {
            'cif_content': sample_cif,
            'atom_indices': [0],
            'new_element': 'N'
        }
    }
    
    for endpoint in endpoints['POST']:
        try:
            response = requests.post(
                f"{base_url}{endpoint}",
                headers={'Content-Type': 'application/json'},
                json=post_data.get(endpoint, {}),
                timeout=10
            )
            results[f"POST {endpoint}"] = {
                'status_code': response.status_code,
                'success': response.status_code in [200, 201],
                'content_type': response.headers.get('content-type', ''),
                'size': len(response.text)
            }
            
            # Try to parse JSON response
            try:
                json_response = response.json()
                results[f"POST {endpoint}"]['json_valid'] = True
                results[f"POST {endpoint}"]['has_success_field'] = 'success' in json_response
            except:
                results[f"POST {endpoint}"]['json_valid'] = False
            
            status = "✅" if results[f"POST {endpoint}"]['success'] else "❌"
            print(f"  {status} POST {endpoint}: {response.status_code}")
            
        except Exception as e:
            results[f"POST {endpoint}"] = {'error': str(e), 'success': False}
            print(f"  ❌ POST {endpoint}: {e}")
    
    return results

def test_frontend_backend_compatibility():
    """Test frontend-backend API compatibility"""
    
    server_host = os.getenv('SERVER_HOST', '127.0.0.1')
    server_port = int(os.getenv('SERVER_PORT', 8080))
    base_url = f"http://{server_host}:{server_port}"
    
    print(f"\n=== Frontend-Backend Compatibility Test ===")
    
    # Common issues to check
    compatibility_issues = []
    
    # 1. Check if frontend expects endpoints that don't exist
    frontend_expected_endpoints = [
        '/parse_cif_enhanced',  # This was in the code
        '/parse_cif',
        '/create_supercell',
        '/delete_atoms',
        '/replace_atoms',
        '/sample_files',
        '/sample_cif/*'
    ]
    
    for endpoint in frontend_expected_endpoints:
        if endpoint == '/sample_cif/*':
            # Test with actual filename
            test_endpoint = '/sample_cif/BaTiO3.cif'
        elif endpoint == '/parse_cif_enhanced':
            # This should not exist
            test_endpoint = endpoint
        else:
            test_endpoint = endpoint
        
        try:
            if endpoint in ['/delete_atoms', '/replace_atoms', '/parse_cif', '/create_supercell']:
                # POST endpoints
                response = requests.post(
                    f"{base_url}{test_endpoint}",
                    headers={'Content-Type': 'application/json'},
                    json={'test': 'data'},
                    timeout=5
                )
            else:
                # GET endpoints
                response = requests.get(f"{base_url}{test_endpoint}", timeout=5)
            
            if response.status_code == 404:
                compatibility_issues.append(f"Endpoint not found: {endpoint}")
            elif response.status_code == 405:
                compatibility_issues.append(f"Method not allowed: {endpoint}")
            
            print(f"  {endpoint}: {response.status_code}")
            
        except Exception as e:
            compatibility_issues.append(f"Cannot reach endpoint {endpoint}: {e}")
            print(f"  ❌ {endpoint}: {e}")
    
    # 2. Check CORS headers
    try:
        response = requests.get(
            f"{base_url}/health",
            headers={'Origin': f'http://localhost:{server_port}'}
        )
        cors_allowed = response.headers.get('Access-Control-Allow-Origin') is not None
        if not cors_allowed:
            compatibility_issues.append("CORS headers missing")
        print(f"  CORS headers: {'✅' if cors_allowed else '❌'}")
    except:
        compatibility_issues.append("Cannot test CORS headers")
    
    return compatibility_issues

def test_browser_simulation():
    """Simulate actual browser requests"""
    
    server_host = os.getenv('SERVER_HOST', '127.0.0.1')
    server_port = int(os.getenv('SERVER_PORT', 8080))
    base_url = f"http://{server_host}:{server_port}"
    
    print(f"\n=== Browser Simulation Test ===")
    
    # Simulate loading the main page
    browser_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Origin': f'http://localhost:{server_port}'
    }
    
    simulation_results = {}
    
    # 1. Load main page
    try:
        response = requests.get(base_url, headers=browser_headers, timeout=10)
        simulation_results['main_page'] = {
            'status_code': response.status_code,
            'success': response.status_code == 200,
            'contains_crystalnexus': 'CrystalNexus' in response.text,
            'contains_js': '<script' in response.text,
            'size': len(response.text)
        }
        print(f"  Main page load: {'✅' if simulation_results['main_page']['success'] else '❌'}")
    except Exception as e:
        simulation_results['main_page'] = {'error': str(e), 'success': False}
        print(f"  ❌ Main page load failed: {e}")
    
    # 2. Simulate health check (like frontend does)
    try:
        api_headers = browser_headers.copy()
        api_headers.update({
            'Accept': 'application/json, text/plain, */*',
            'Content-Type': 'application/json'
        })
        
        response = requests.get(f"{base_url}/health", headers=api_headers, timeout=5)
        simulation_results['health_check'] = {
            'status_code': response.status_code,
            'success': response.status_code == 200,
            'cors_allowed': response.headers.get('Access-Control-Allow-Origin') is not None
        }
        print(f"  Health check: {'✅' if simulation_results['health_check']['success'] else '❌'}")
        
    except Exception as e:
        simulation_results['health_check'] = {'error': str(e), 'success': False}
        print(f"  ❌ Health check failed: {e}")
    
    # 3. Simulate sample file loading
    try:
        response = requests.get(f"{base_url}/sample_files", headers=api_headers, timeout=5)
        simulation_results['sample_files'] = {
            'status_code': response.status_code,
            'success': response.status_code == 200,
            'cors_allowed': response.headers.get('Access-Control-Allow-Origin') is not None
        }
        print(f"  Sample files: {'✅' if simulation_results['sample_files']['success'] else '❌'}")
        
    except Exception as e:
        simulation_results['sample_files'] = {'error': str(e), 'success': False}
        print(f"  ❌ Sample files failed: {e}")
    
    return simulation_results

def main():
    """Run comprehensive endpoint connectivity tests"""
    print("Enhanced Endpoint Connectivity Test Suite")
    print("=" * 60)
    
    try:
        # Run all tests
        endpoint_results = test_all_endpoints()
        compatibility_issues = test_frontend_backend_compatibility()
        browser_results = test_browser_simulation()
        
        # Summary
        print(f"\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        
        # Endpoint test summary
        successful_endpoints = sum(1 for result in endpoint_results.values() if result.get('success', False))
        total_endpoints = len(endpoint_results)
        print(f"Endpoint Tests:  {successful_endpoints}/{total_endpoints} endpoints working")
        
        # Compatibility summary
        print(f"Compatibility:   {'✅ NO ISSUES' if not compatibility_issues else f'❌ {len(compatibility_issues)} ISSUES'}")
        if compatibility_issues:
            for issue in compatibility_issues:
                print(f"  • {issue}")
        
        # Browser simulation summary
        browser_success = all(result.get('success', False) for result in browser_results.values())
        print(f"Browser Sim:     {'✅ PASS' if browser_success else '❌ FAIL'}")
        
        # Overall result
        overall_success = (
            successful_endpoints == total_endpoints and
            not compatibility_issues and
            browser_success
        )
        
        print(f"\nOVERALL RESULT:  {'✅ ALL SYSTEMS OPERATIONAL' if overall_success else '❌ CONNECTIVITY ISSUES DETECTED'}")
        
        if not overall_success:
            print("\n🔧 Recommended Actions:")
            if successful_endpoints < total_endpoints:
                print("  1. Check server logs for endpoint errors")
            if compatibility_issues:
                print("  2. Fix frontend-backend API mismatches")
            if not browser_success:
                print("  3. Check CORS configuration and server availability")
                
        return overall_success
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)