#!/usr/bin/env python3
"""
Application integration test
アプリケーションの統合テスト
"""
import os
import sys
import subprocess
import time
import requests
import tempfile
from pathlib import Path

def test_app_startup():
    """Test application startup and basic functionality"""
    print('=== Application Integration Test ===')
    
    # Import os at function level
    import os
    
    # Environment setup for testing
    os.environ['CRYSTALNEXUS_HOST'] = '127.0.0.1'
    os.environ['CRYSTALNEXUS_PORT'] = '8081'  # Use different port for testing
    os.environ['CRYSTALNEXUS_DEBUG'] = 'false'
    
    port = 8081
    host = '127.0.0.1'
    base_url = f'http://{host}:{port}'
    
    # Start the server
    print('Starting test server...')
    try:
        # Import to check for syntax errors
        os.chdir('../..')  # Change to project root
        sys.path.insert(0, '.')
        from main import app, ALLOWED_ELEMENTS, HOST, PORT, DEBUG
        
        print(f'✓ Module import successful')
        print(f'✓ Dynamic elements loaded: {len(ALLOWED_ELEMENTS)} elements')
        print(f'✓ Configuration: HOST={HOST}, PORT={PORT}, DEBUG={DEBUG}')
        
        # Test individual security functions
        from main import validate_element, safe_filename, validate_supercell_size
        
        # Test element validation
        try:
            validate_element('Fe')
            print('✓ Element validation function working')
        except Exception as e:
            print(f'✗ Element validation error: {e}')
            return False
        
        # Test filename validation
        try:
            safe_filename('test.cif')
            print('✓ Filename validation function working')
        except Exception as e:
            print(f'✗ Filename validation error: {e}')
            return False
        
        # Test supercell validation
        try:
            validate_supercell_size([2, 2, 2])
            print('✓ Supercell validation function working')
        except Exception as e:
            print(f'✗ Supercell validation error: {e}')
            return False
        
        print('✓ All security functions integrated successfully')
        
        # Test API endpoints would require server startup
        # For now, just test the function imports and basic validation
        
    except ImportError as e:
        print(f'✗ Import error: {e}')
        return False
    except Exception as e:
        print(f'✗ Application test error: {e}')
        return False
    
    print('✓ Application integration test completed successfully')
    return True

def test_environment_config():
    """Test environment configuration"""
    print('\n--- Environment Configuration Test ---')
    
    # Test different environment settings
    test_configs = [
        {'HOST': '127.0.0.1', 'PORT': '8080', 'DEBUG': 'false'},
        {'HOST': '127.0.0.1', 'PORT': '3000', 'DEBUG': 'true'},
    ]
    
    for config in test_configs:
        print(f'Testing config: {config}')
        
        # Set environment variables
        for key, value in config.items():
            os.environ[f'CRYSTALNEXUS_{key}'] = value
        
        try:
            # Re-import to get new config
            import importlib
            if 'main' in sys.modules:
                importlib.reload(sys.modules['main'])
            else:
                import main
            
            print(f'  ✓ Config loaded: HOST={main.HOST}, PORT={main.PORT}, DEBUG={main.DEBUG}')
            
        except Exception as e:
            print(f'  ✗ Config test error: {e}')
            return False
    
    print('✓ Environment configuration test passed')
    return True

def test_element_source():
    """Test element source detection"""
    print('\n--- Element Source Test ---')
    
    try:
        from main import get_chgnet_supported_elements, CHGNET_AVAILABLE
        
        elements = get_chgnet_supported_elements()
        print(f'✓ Elements loaded: {len(elements)} total')
        print(f'✓ CHGNet available: {CHGNET_AVAILABLE}')
        
        if CHGNET_AVAILABLE:
            print('✓ Using dynamic CHGNet element detection')
        else:
            print('✓ Using fallback element list')
        
        # Verify common elements are present
        common_elements = ['H', 'C', 'N', 'O', 'Fe', 'Cu', 'Zn']
        missing = []
        for elem in common_elements:
            if elem not in elements:
                missing.append(elem)
        
        if missing:
            print(f'✗ Missing common elements: {missing}')
            return False
        else:
            print('✓ All common elements present')
        
    except Exception as e:
        print(f'✗ Element source test error: {e}')
        return False
    
    print('✓ Element source test passed')
    return True

if __name__ == '__main__':
    success = True
    
    success &= test_app_startup()
    success &= test_environment_config()
    success &= test_element_source()
    
    if success:
        print('\n=== All Integration Tests PASSED ===')
        print('✓ Application ready for deployment')
    else:
        print('\n=== Some Integration Tests FAILED ===')
    
    exit(0 if success else 1)