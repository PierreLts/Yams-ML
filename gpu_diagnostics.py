import os
import sys
import subprocess
import platform
import ctypes
from ctypes.util import find_library

def print_section(title):
    print("\n" + "="*50)
    print(f" {title} ")
    print("="*50)

def check_command(command):
    """Run a command and return its output."""
    try:
        result = subprocess.run(command, shell=True, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                               universal_newlines=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr.strip()}"

def check_tf_gpu():
    """Check if TensorFlow can access the GPU."""
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        # Check if TensorFlow can see the GPU
        physical_devices = tf.config.list_physical_devices('GPU')
        print(f"GPU devices detected by TensorFlow: {physical_devices}")
        
        # Check if CUDA is built into TensorFlow
        print(f"CUDA built into TensorFlow: {tf.test.is_built_with_cuda()}")
        
        # Try to get GPU device name
        if len(physical_devices) > 0:
            try:
                with tf.device('/GPU:0'):
                    a = tf.constant([[1.0, 2.0]], dtype=tf.float32)
                    b = tf.constant([[3.0], [4.0]], dtype=tf.float32)
                    c = tf.matmul(a, b)
                    print(f"Matrix multiplication result: {c}")
                    print("✅ TensorFlow successfully executed operations on GPU!")
            except Exception as e:
                print(f"❌ Error executing operations on GPU: {e}")
        
        return len(physical_devices) > 0
    except ImportError:
        print("TensorFlow is not installed.")
        return False
    except Exception as e:
        print(f"Error checking TensorFlow GPU: {e}")
        return False

def check_pytorch_gpu():
    """Check if PyTorch can access the GPU as a comparison."""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available in PyTorch: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device name: {torch.cuda.get_device_name(0)}")
            
            try:
                a = torch.tensor([[1.0, 2.0]], device='cuda')
                b = torch.tensor([[3.0], [4.0]], device='cuda')
                c = torch.matmul(a, b)
                print(f"Matrix multiplication result: {c}")
                print("✅ PyTorch successfully executed operations on GPU!")
            except Exception as e:
                print(f"❌ Error executing operations on GPU: {e}")
            
            return True
        return False
    except ImportError:
        print("PyTorch is not installed.")
        return False
    except Exception as e:
        print(f"Error checking PyTorch GPU: {e}")
        return False

def check_cuda_paths():
    """Check CUDA paths and DLLs."""
    # First check environment variables
    cuda_path = os.environ.get('CUDA_PATH', '')
    cuda_home = os.environ.get('CUDA_HOME', '')
    
    print(f"CUDA_PATH environment variable: {cuda_path or 'Not set'}")
    print(f"CUDA_HOME environment variable: {cuda_home or 'Not set'}")
    
    # Check system PATH
    path_env = os.environ.get('PATH', '')
    print("\nCUDA directories in PATH:")
    cuda_in_path = False
    cudnn_in_path = False
    
    for p in path_env.split(os.pathsep):
        if 'cuda' in p.lower():
            print(f" - {p}")
            cuda_in_path = True
            
            # Check for cudnn in this directory
            if os.path.exists(os.path.join(p, 'cudnn64_8.dll')) or os.path.exists(os.path.join(p, 'cudnn.dll')):
                cudnn_in_path = True
                print(f"   ✅ Found cuDNN in this directory")
    
    if not cuda_in_path:
        print(" - No CUDA directories found in PATH")
    
    # Try to locate CUDA installation
    cuda_dirs = []
    potential_locations = [
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
        "C:\\Program Files\\NVIDIA\\CUDA"
    ]
    
    for location in potential_locations:
        if os.path.exists(location):
            for d in os.listdir(location):
                if d.startswith('v'):
                    cuda_dir = os.path.join(location, d)
                    if os.path.isdir(cuda_dir):
                        cuda_dirs.append(cuda_dir)
    
    if cuda_dirs:
        print("\nFound CUDA installations:")
        for d in cuda_dirs:
            print(f" - {d}")
    else:
        print("\nNo CUDA installations found in standard locations")
    
    # Check for cuDNN
    print("\nSearching for cuDNN:")
    found_cudnn = False
    
    # First check standard CUDA directories
    for cuda_dir in cuda_dirs:
        cudnn_paths = [
            os.path.join(cuda_dir, "bin", "cudnn64_8.dll"),
            os.path.join(cuda_dir, "bin", "cudnn64_9.dll"),
            os.path.join(cuda_dir, "bin", "cudnn.dll")
        ]
        
        for path in cudnn_paths:
            if os.path.exists(path):
                print(f" ✅ Found cuDNN: {path}")
                found_cudnn = True
    
    # Check Windows system directories
    system_dirs = [
        os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'System32'),
        os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'SysWOW64')
    ]
    
    for sys_dir in system_dirs:
        cudnn_paths = [
            os.path.join(sys_dir, "cudnn64_8.dll"),
            os.path.join(sys_dir, "cudnn64_9.dll"),
            os.path.join(sys_dir, "cudnn.dll")
        ]
        
        for path in cudnn_paths:
            if os.path.exists(path):
                print(f" ✅ Found cuDNN in system directory: {path}")
                found_cudnn = True
    
    # Try to load cuDNN using ctypes
    try:
        cudnn = ctypes.CDLL("cudnn64_8.dll")
        print(" ✅ Successfully loaded cuDNN library!")
        found_cudnn = True
    except OSError:
        try:
            cudnn = ctypes.CDLL("cudnn64_9.dll")
            print(" ✅ Successfully loaded cuDNN v9 library!")
            found_cudnn = True
        except OSError:
            try:
                cudnn = ctypes.CDLL("cudnn.dll")
                print(" ✅ Successfully loaded generic cuDNN library!")
                found_cudnn = True
            except OSError:
                print(" ❌ Could not load cuDNN library dynamically")
    
    if not found_cudnn:
        print(" ❌ cuDNN NOT found! This is likely why TensorFlow isn't detecting your GPU.")
    
    return found_cudnn, cuda_in_path

def main():
    print_section("SYSTEM INFORMATION")
    print(f"Python version: {platform.python_version()}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    
    print_section("NVIDIA GPU CHECK")
    nvidia_smi = check_command("nvidia-smi")
    print(nvidia_smi)
    
    print_section("NVCC (CUDA Compiler) CHECK")
    nvcc_version = check_command("nvcc --version")
    print(nvcc_version)
    
    print_section("CUDA & cuDNN PATHS")
    found_cudnn, cuda_in_path = check_cuda_paths()
    
    print_section("TENSORFLOW GPU CHECK")
    tf_gpu_works = check_tf_gpu()
    
    # Optionally check PyTorch if TensorFlow doesn't work
    if not tf_gpu_works:
        try:
            import torch
            print_section("PYTORCH GPU CHECK (ALTERNATIVE)")
            pytorch_gpu_works = check_pytorch_gpu()
        except ImportError:
            pytorch_gpu_works = False
            print("PyTorch not installed, skipping PyTorch GPU check")
    
    print_section("DIAGNOSTICS SUMMARY")
    
    if not cuda_in_path:
        print("❌ CUDA not found in PATH. Add CUDA bin directory to your PATH.")
    
    if not found_cudnn:
        print("❌ cuDNN not found. Install cuDNN or check its installation.")
    
    if tf_gpu_works:
        print("✅ TensorFlow GPU acceleration is working!")
    else:
        print("❌ TensorFlow GPU acceleration is NOT working.")
        
        # Recommendations based on findings
        print("\nRECOMMENDED ACTIONS:")
        
        if not cuda_in_path:
            print("1. Add CUDA bin directory to your PATH environment variable:")
            print("   - Right-click on This PC → Properties → Advanced system settings")
            print("   - Click on Environment Variables → System variables → Path → Edit")
            print("   - Add 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin'")
        
        if not found_cudnn:
            print("2. Ensure cuDNN is properly installed:")
            print("   - The installer you used should have placed cuDNN files in the CUDA directories")
            print("   - Verify cudnn*.dll files exist in CUDA/bin directory")
        
        print("3. Try installing PyTorch as an alternative:")
        print("   - Create a new environment: python -m venv .venv_torch")
        print("   - Install PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
        print("4. Try reinstalling TensorFlow:")
        print("   - pip uninstall tensorflow")
        print("   - pip install tensorflow==2.12.0")

if __name__ == "__main__":
    main()