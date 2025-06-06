#!/usr/bin/env python3
"""
Robust CUDA functionality test that handles different package versions and attribute access.
"""

import warnings
warnings.filterwarnings('ignore')

def safe_get_version(module, module_name):
    """Safely get version from a module."""
    version_attrs = ['__version__', 'version', '_version', 'VERSION']
    
    for attr in version_attrs:
        try:
            version = getattr(module, attr, None)
            if version:
                return str(version)
        except:
            continue
    
    # Try to get version from package metadata
    try:
        import importlib.metadata
        return importlib.metadata.version(module_name)
    except:
        pass
    
    return "Version unavailable"

def test_cupy():
    """Test CuPy functionality."""
    print("🧪 Testing CuPy...")
    
    try:
        import cupy as cp
        print("✅ CuPy imported successfully")
        
        # Get version safely
        version = safe_get_version(cp, 'cupy')
        print(f"   Version: {version}")
        
        # Test basic array operations
        try:
            x = cp.array([1, 2, 3, 4, 5])
            result = cp.sum(x)
            print(f"   ✅ Basic operation: sum([1,2,3,4,5]) = {result}")
        except Exception as e:
            print(f"   ❌ Basic operation failed: {e}")
            return False
        
        # Test more complex operations
        try:
            a = cp.random.random((3, 3))
            b = cp.random.random((3, 3))
            c = cp.dot(a, b)
            print(f"   ✅ Matrix multiplication: {c.shape} result")
        except Exception as e:
            print(f"   ⚠️  Matrix operation failed: {e}")
        
        # Check CUDA runtime (safely)
        try:
            runtime_version = cp.cuda.runtime.runtimeGetVersion()
            print(f"   CUDA Runtime: {runtime_version}")
        except Exception as e:
            print(f"   ⚠️  CUDA Runtime unavailable: {e}")
        
        # Check device count (safely)
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            print(f"   GPU Devices detected: {device_count}")
            
            if device_count > 0:
                for i in range(device_count):
                    try:
                        device = cp.cuda.Device(i)
                        with device:
                            mem_info = cp.cuda.MemoryInfo()
                            total_gb = mem_info.total / 1024**3
                            print(f"   GPU {i}: {total_gb:.1f}GB memory")
                    except Exception as e:
                        print(f"   GPU {i}: Info unavailable ({e})")
            else:
                print("   ⚠️  No GPU devices detected (CPU fallback mode)")
                
        except Exception as e:
            print(f"   ⚠️  GPU detection failed: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ CuPy import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ CuPy test failed: {e}")
        return False

def test_cudf():
    """Test cuDF functionality."""
    print("\n🧪 Testing cuDF...")
    
    try:
        import cudf
        print("✅ cuDF imported successfully")
        
        # Get version safely
        version = safe_get_version(cudf, 'cudf')
        print(f"   Version: {version}")
        
        # Test basic DataFrame operations
        try:
            df = cudf.DataFrame({
                'a': [1, 2, 3, 4, 5],
                'b': [2.0, 3.0, 4.0, 5.0, 6.0],
                'c': ['x', 'y', 'z', 'w', 'v']
            })
            
            # Test various operations
            mean_a = df['a'].mean()
            sum_b = df['b'].sum()
            count_c = df['c'].count()
            
            print(f"   ✅ DataFrame operations: mean={mean_a}, sum={sum_b:.1f}, count={count_c}")
            
            # Test groupby
            df_grouped = df.groupby('c')['a'].sum()
            print(f"   ✅ GroupBy operation: {len(df_grouped)} groups")
            
            return True
            
        except Exception as e:
            print(f"   ❌ DataFrame operations failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ cuDF import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ cuDF test failed: {e}")
        return False

def test_cuml():
    """Test cuML functionality."""
    print("\n🧪 Testing cuML...")
    
    try:
        import cuml
        print("✅ cuML imported successfully")
        
        # Get version safely
        version = safe_get_version(cuml, 'cuml')
        print(f"   Version: {version}")
        
        # Test basic ML operations
        try:
            # Try to import and test a simple algorithm
            from cuml.cluster import KMeans
            from cuml.datasets import make_blobs
            
            # Generate sample data
            X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
            
            # Fit KMeans
            kmeans = KMeans(n_clusters=3, random_state=42)
            kmeans.fit(X)
            
            print("   ✅ KMeans clustering: Successfully trained on sample data")
            print(f"   ✅ Cluster centers shape: {kmeans.cluster_centers_.shape}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ ML operations failed: {e}")
            print("   ℹ️  cuML imported but ML operations unavailable")
            return False
            
    except ImportError as e:
        print(f"❌ cuML import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ cuML test failed: {e}")
        return False

def test_numpy_compatibility():
    """Test NumPy compatibility."""
    print("\n🧪 Testing NumPy compatibility...")
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
        
        # Get version safely
        version = safe_get_version(np, 'numpy')
        print(f"   Version: {version}")
        
        # Test basic operations
        x = np.array([1, 2, 3, 4, 5])
        result = np.sum(x)
        print(f"   ✅ Basic operation: sum([1,2,3,4,5]) = {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ NumPy test failed: {e}")
        return False

def main():
    """Run all tests and provide summary."""
    print("🚀 Comprehensive CUDA Package Test")
    print("=" * 50)
    
    # Run all tests
    test_results = {
        'numpy': test_numpy_compatibility(),
        'cupy': test_cupy(),
        'cudf': test_cudf(),
        'cuml': test_cuml()
    }
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    working_packages = []
    failing_packages = []
    
    for package, result in test_results.items():
        if result:
            working_packages.append(package)
            print(f"✅ {package}: Working")
        else:
            failing_packages.append(package)
            print(f"❌ {package}: Issues detected")
    
    print(f"\n📈 Success Rate: {len(working_packages)}/{len(test_results)} packages working")
    
    # Provide recommendations
    print("\n💡 Recommendations:")
    print("-" * 30)
    
    if len(working_packages) >= 3:
        print("🎉 Excellent! Most CUDA packages are working.")
        print("✅ You can proceed with GPU-accelerated processing:")
        print("   python analyze.py preprocess-gpu --verify-cuda")
        
    elif len(working_packages) >= 2:
        print("✅ Good! Core packages are working.")
        print("✅ Recommended approach:")
        print("   python analyze.py preprocess-gpu --fallback-cpu")
        
    elif len(working_packages) >= 1:
        print("⚠️  Some packages working, but issues detected.")
        print("💡 Recommended approach:")
        print("   python analyze.py preprocess --fallback-cpu")
        
    else:
        print("❌ Significant issues detected.")
        print("🔧 Try reinstalling packages:")
        print("   pip uninstall cupy-cuda12x cudf cuml -y")
        print("   pip install cupy-cuda12x")
        print("   conda install -c rapidsai cudf cuml")
    
    # Check environment info
    print("\n🔍 Environment Information:")
    print("-" * 30)
    
    try:
        import sys
        print(f"Python: {sys.version.split()[0]}")
        print(f"Platform: {sys.platform}")
        
        # Check if we're in WSL
        try:
            with open('/proc/version', 'r') as f:
                if 'microsoft' in f.read().lower():
                    print("Environment: WSL (Windows Subsystem for Linux)")
                else:
                    print("Environment: Native Linux")
        except:
            print("Environment: Unknown")
            
    except:
        print("Environment info unavailable")
    
    return len(working_packages) >= 2

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)