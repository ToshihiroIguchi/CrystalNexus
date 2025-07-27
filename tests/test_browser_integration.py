#!/usr/bin/env python3
"""
Browser Integration Test - CORS and Frontend Connectivity
This test detects CORS issues and browser-specific connection problems
"""

import requests
import json
import time
import os
from urllib.parse import urlparse

def test_cors_configuration():
    """Test CORS configuration for browser compatibility"""
    
    # Get server info from environment or default
    server_host = os.getenv('SERVER_HOST', '127.0.0.1')
    server_port = int(os.getenv('SERVER_PORT', 8080))
    
    base_url = f"http://{server_host}:{server_port}"
    
    print("=== CORS Configuration Test ===")
    print(f"Testing server: {base_url}")
    
    # Test different origin scenarios
    test_origins = [
        f"http://localhost:{server_port}",
        f"http://127.0.0.1:{server_port}",
        f"http://{server_host}:{server_port}",
        "http://localhost:3000",  # Common dev port
        "http://localhost:5000",  # Default Flask port
    ]
    
    cors_results = {}
    
    for origin in test_origins:
        print(f"\nTesting origin: {origin}")
        
        try:
            # Test preflight request (OPTIONS)
            options_response = requests.options(
                f"{base_url}/parse_cif",
                headers={
                    'Origin': origin,
                    'Access-Control-Request-Method': 'POST',
                    'Access-Control-Request-Headers': 'Content-Type'
                },
                timeout=5
            )
            
            # Test actual request with Origin header
            test_response = requests.get(
                f"{base_url}/health",
                headers={'Origin': origin},
                timeout=5
            )
            
            cors_headers = {
                'access-control-allow-origin': test_response.headers.get('Access-Control-Allow-Origin'),
                'access-control-allow-methods': test_response.headers.get('Access-Control-Allow-Methods'),
                'access-control-allow-headers': test_response.headers.get('Access-Control-Allow-Headers'),
            }
            
            cors_results[origin] = {
                'status_code': test_response.status_code,
                'cors_headers': cors_headers,
                'allowed': cors_headers['access-control-allow-origin'] is not None
            }
            
            if cors_headers['access-control-allow-origin']:
                print(f"  ✅ CORS allowed - Origin: {cors_headers['access-control-allow-origin']}")
            else:
                print(f"  ❌ CORS blocked - No Access-Control-Allow-Origin header")
                
        except Exception as e:
            print(f"  ❌ Request failed: {e}")
            cors_results[origin] = {
                'status_code': None,
                'cors_headers': {},
                'allowed': False,
                'error': str(e)
            }
    
    return cors_results

def test_html_serving():
    """Test if HTML is served correctly"""
    
    server_host = os.getenv('SERVER_HOST', '127.0.0.1')
    server_port = int(os.getenv('SERVER_PORT', 8080))
    base_url = f"http://{server_host}:{server_port}"
    
    print(f"\n=== HTML Serving Test ===")
    
    try:
        response = requests.get(base_url, timeout=10)
        
        if response.status_code == 200:
            html_content = response.text
            
            # Check for key elements
            checks = {
                'html_structure': '<html' in html_content,
                'title_present': 'CrystalNexus' in html_content,
                'javascript_present': '<script' in html_content,
                'api_calls_present': 'fetch(' in html_content or 'XMLHttpRequest' in html_content,
                'server_url_config': f':{server_port}' in html_content
            }
            
            print(f"  HTML response status: {response.status_code}")
            print(f"  Content length: {len(html_content)} chars")
            
            for check_name, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"  {status} {check_name.replace('_', ' ').title()}: {passed}")
            
            return {
                'status_code': response.status_code,
                'content_length': len(html_content),
                'checks': checks,
                'all_checks_passed': all(checks.values())
            }
            
        else:
            print(f"  ❌ HTML serving failed: {response.status_code}")
            return {
                'status_code': response.status_code,
                'content_length': 0,
                'checks': {},
                'all_checks_passed': False
            }
            
    except Exception as e:
        print(f"  ❌ HTML serving error: {e}")
        return {
            'status_code': None,
            'content_length': 0,
            'checks': {},
            'all_checks_passed': False,
            'error': str(e)
        }

def test_api_endpoints_from_browser_perspective():
    """Test API endpoints as a browser would"""
    
    server_host = os.getenv('SERVER_HOST', '127.0.0.1')
    server_port = int(os.getenv('SERVER_PORT', 8080))
    base_url = f"http://{server_host}:{server_port}"
    origin = f"http://localhost:{server_port}"
    
    print(f"\n=== Browser Perspective API Test ===")
    print(f"Simulating requests from origin: {origin}")
    
    # Browser-like headers
    browser_headers = {
        'Origin': origin,
        'Referer': f"{origin}/",
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Content-Type': 'application/json',
    }
    
    test_results = {}
    
    # Test health endpoint
    try:
        health_response = requests.get(
            f"{base_url}/health",
            headers={k: v for k, v in browser_headers.items() if k != 'Content-Type'},
            timeout=5
        )
        
        cors_allowed = health_response.headers.get('Access-Control-Allow-Origin') is not None
        
        test_results['health'] = {
            'status_code': health_response.status_code,
            'cors_allowed': cors_allowed,
            'response_size': len(health_response.text)
        }
        
        print(f"  Health endpoint: {health_response.status_code}")
        print(f"  CORS allowed: {'✅' if cors_allowed else '❌'}")
        
    except Exception as e:
        test_results['health'] = {'error': str(e)}
        print(f"  ❌ Health endpoint failed: {e}")
    
    # Test CIF parsing endpoint with sample data
    try:
        sample_cif = """data_sample
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
        
        parse_response = requests.post(
            f"{base_url}/parse_cif",
            headers=browser_headers,
            json={'cif_content': sample_cif},
            timeout=10
        )
        
        cors_allowed = parse_response.headers.get('Access-Control-Allow-Origin') is not None
        
        test_results['parse_cif'] = {
            'status_code': parse_response.status_code,
            'cors_allowed': cors_allowed,
            'response_size': len(parse_response.text)
        }
        
        print(f"  Parse CIF endpoint: {parse_response.status_code}")
        print(f"  CORS allowed: {'✅' if cors_allowed else '❌'}")
        
    except Exception as e:
        test_results['parse_cif'] = {'error': str(e)}
        print(f"  ❌ Parse CIF endpoint failed: {e}")
    
    return test_results

def analyze_cors_compatibility():
    """Analyze CORS compatibility and provide recommendations"""
    
    server_host = os.getenv('SERVER_HOST', '127.0.0.1')
    server_port = int(os.getenv('SERVER_PORT', 8080))
    
    print(f"\n=== CORS Compatibility Analysis ===")
    
    # Expected origins for this port
    expected_origins = [
        f"http://localhost:{server_port}",
        f"http://127.0.0.1:{server_port}",
        f"http://{server_host}:{server_port}"
    ]
    
    # Test each expected origin
    compatibility_issues = []
    
    for origin in expected_origins:
        try:
            response = requests.get(
                f"http://{server_host}:{server_port}/health",
                headers={'Origin': origin},
                timeout=5
            )
            
            allowed_origin = response.headers.get('Access-Control-Allow-Origin')
            
            if not allowed_origin:
                compatibility_issues.append(f"No CORS headers for origin: {origin}")
            elif allowed_origin != origin and allowed_origin != '*':
                compatibility_issues.append(f"Origin mismatch - Expected: {origin}, Got: {allowed_origin}")
                
        except Exception as e:
            compatibility_issues.append(f"Failed to test origin {origin}: {e}")
    
    print(f"Port being tested: {server_port}")
    print(f"Expected origins: {expected_origins}")
    
    if compatibility_issues:
        print("\n❌ CORS Compatibility Issues Found:")
        for issue in compatibility_issues:
            print(f"  • {issue}")
        
        print(f"\n💡 Recommendations:")
        print(f"  1. Update CORS allowed_origins to include port {server_port}")
        print(f"  2. Add these origins to the server configuration:")
        for origin in expected_origins:
            print(f"     - {origin}")
        print(f"  3. Consider using environment-based CORS configuration")
        
        return False
    else:
        print("✅ CORS configuration is compatible")
        return True

def main():
    """Run comprehensive browser integration tests"""
    print("Browser Integration Test Suite")
    print("=" * 50)
    
    try:
        # Run all tests
        cors_results = test_cors_configuration()
        html_results = test_html_serving()
        api_results = test_api_endpoints_from_browser_perspective()
        cors_compatible = analyze_cors_compatibility()
        
        # Summary
        print(f"\n" + "=" * 50)
        print("TEST RESULTS SUMMARY")
        print("=" * 50)
        
        # CORS test summary
        allowed_origins = sum(1 for result in cors_results.values() if result.get('allowed', False))
        total_origins = len(cors_results)
        print(f"CORS Tests:      {allowed_origins}/{total_origins} origins allowed")
        
        # HTML serving summary
        html_ok = html_results.get('all_checks_passed', False)
        print(f"HTML Serving:    {'✅ PASS' if html_ok else '❌ FAIL'}")
        
        # API tests summary
        api_health_ok = api_results.get('health', {}).get('cors_allowed', False)
        api_parse_ok = api_results.get('parse_cif', {}).get('cors_allowed', False)
        print(f"API CORS:        Health: {'✅' if api_health_ok else '❌'}, Parse: {'✅' if api_parse_ok else '❌'}")
        
        # Overall compatibility
        print(f"CORS Compatible: {'✅ YES' if cors_compatible else '❌ NO'}")
        
        # Overall result
        overall_success = html_ok and api_health_ok and api_parse_ok and cors_compatible
        print(f"\nOVERALL RESULT:  {'✅ BROWSER COMPATIBLE' if overall_success else '❌ BROWSER INTEGRATION ISSUES'}")
        
        if not overall_success:
            print("\n🚨 Browser integration issues detected!")
            print("This explains why the browser shows 'Cannot connect to server'")
            
        return overall_success
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)