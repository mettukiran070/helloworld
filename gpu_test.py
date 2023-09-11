import torch

for i in range(100):
    print(i)
    # Check if a GPU is available and set the device accordingly
    device = torch.device("cuda:0")
    print(device)

    # Define matrix sizes for the multiplication
    matrix_size = 100
    a = torch.randn(matrix_size, matrix_size).to(device)
    b = torch.randn(matrix_size, matrix_size).to(device)

    # Perform matrix multiplication on the GPU
    result = torch.matmul(a, b)
    # Ensure the GPU memory is freed after use
    torch.cuda.empty_cache()

    print(result)
    
print("Closed")
