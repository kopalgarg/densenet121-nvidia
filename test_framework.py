#!/usr/bin/env python3
"""
🧪 Test Script for Medical Imaging Pipeline Analysis Framework
This script tests the basic functionality of the framework
"""

import os
import sys
import importlib

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing module imports...")
    
    required_modules = [
        'torch',
        'numpy',
        'matplotlib',
        'seaborn',
        'PIL',
        'cv2',
        'skimage',
        'scipy',
        'pandas'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️  Failed imports: {failed_imports}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All required modules imported successfully!")
    return True

def test_config():
    """Test configuration loading"""
    print("\n🧪 Testing configuration...")
    
    try:
        from config import validate_config
        validate_config()
        print("✅ Configuration loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_analysis_scripts():
    """Test if analysis scripts exist and are valid Python"""
    print("\n🧪 Testing analysis scripts...")
    
    analysis_scripts = [
        'comprehensive_gpu_analysis.py',
        'comprehensive_cpu_analysis.py',
        'enhanced_gpu_cpu_comparison.py',
        'download_mednist.py'
    ]
    
    failed_scripts = []
    
    for script in analysis_scripts:
        if os.path.exists(script):
            try:
                with open(script, 'r') as f:
                    content = f.read()
                    compile(content, script, 'exec')
                print(f"✅ {script}")
            except SyntaxError as e:
                print(f"❌ {script}: Syntax error - {e}")
                failed_scripts.append(script)
        else:
            print(f"❌ {script}: File not found")
            failed_scripts.append(script)
    
    if failed_scripts:
        print(f"\n⚠️  Failed scripts: {failed_scripts}")
        return False
    
    print("✅ All analysis scripts are valid Python!")
    return True

def test_directory_structure():
    """Test if required directories exist"""
    print("\n🧪 Testing directory structure...")
    
    required_dirs = [
        'analysis',
        'data',
        'models',
        'utils',
        'mednist'
    ]
    
    missing_dirs = []
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/: Directory not found")
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"\n⚠️  Missing directories: {missing_dirs}")
        return False
    
    print("✅ All required directories exist!")
    return True

def main():
    """Main test function"""
    print("🏥 Medical Imaging Pipeline Analysis Framework - Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_config,
        test_analysis_scripts,
        test_directory_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("📊 Test Results:")
    print(f"✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Framework is ready to use.")
        print("\n🚀 Next steps:")
        print("1. Download MedNIST: python download_mednist.py")
        print("2. Run GPU analysis: python comprehensive_gpu_analysis.py")
        print("3. Run CPU analysis: python comprehensive_cpu_analysis.py")
        print("4. Run comparison: python enhanced_gpu_cpu_comparison.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
