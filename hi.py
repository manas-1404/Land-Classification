import tensorflow as tf
import os

print(f"TensorFlow Version: {tf.__version__}")

# Check for available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print("GPUs: ", gpus)
    for gpu in gpus:
        print(f"Device details: {gpu}")
else:
    print("No GPUs detected")

# Use nvidia-smi to check CUDA version
os.system('nvidia-smi')

# Verify CUDA_PATH environment variable
cuda_path = os.environ.get('CUDA_PATH')
if cuda_path:
    print(f"CUDA_PATH is set to: {cuda_path}")
else:
    print("CUDA_PATH is not set")

# Check for cuDNN DLL existence
cudnn_dll = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.3\\bin\\cudnn64_8.dll"
if os.path.isfile(cudnn_dll):
    print(f"cuDNN DLL found at: {cudnn_dll}")
else:
    print("cuDNN DLL not found")

# Additional TensorFlow GPU configuration checks
try:
    tf.debugging.set_log_device_placement(True)
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        c = tf.matmul(a, b)
        print(c)
except RuntimeError as e:
    print(f"Error during TensorFlow GPU operation: {e}")
