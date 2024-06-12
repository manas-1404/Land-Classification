import numpy as np

# Replace 'your_file.npz' with the path to your npz file
file_path = r'C:\Users\lyq09mow\Code\Land-Classification\madrid-es\ground_truth_class_raster_50km.npz'

# Load the npz file
data = np.load(file_path)

# Iterate through the contents and print them
for key in data.files:
    print(f'Array name: {key}')
    print("-"*50)
    print(f'Array contents:\n{data[key]}\n')

# Close the npz file
data.close()
