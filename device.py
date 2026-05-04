import torch
import triton.runtime.driver as driver
import torch_npu

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]
aicore_num = properties["num_aicore"]

print(f"Current NPU device: {device}")
print(f"Number of VectorCores: {vectorcore_num}")
print(f"Number of AICores: {aicore_num}")
# Current NPU device: 0
# Number of VectorCores: 40
# Number of AICores: 20