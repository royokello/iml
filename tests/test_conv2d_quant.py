import torch
import sys
import os
sys.path.append(os.getcwd())
from models.blocks.conv_2d import QuantConv1x1
import math
import torch.nn.functional as F

def test_conv2d_block_quant():
    print("Testing QuantConv1x1...")
    in_channels = 64
    out_channels = 128
    kernel_size = 1
    
    # 1. Instantiate
    block = QuantConv1x1(in_channels, out_channels, bias=True)
    
    # 2. Create random data
    # Int8 weights
    weight_int8 = torch.randint(-127, 127, (out_channels, in_channels, kernel_size, kernel_size), dtype=torch.int8)
    
    # Scales
    total_elements = out_channels * in_channels * kernel_size * kernel_size
    num_blocks = math.ceil(total_elements / 32)
    weight_scale = torch.rand(num_blocks, dtype=torch.float16)
    
    print(f"Weight shape: {weight_int8.shape}")
    print(f"Scale shape: {weight_scale.shape}")
    
    # 3. Load weights
    block.load_quantized_weights(weight_int8, weight_scale)
    
    # 4. Input
    x = torch.randn(1, in_channels, 32, 32, dtype=torch.float16)
    
    # 5. Run forward
    with torch.no_grad():
        y = block(x)
        
    print(f"Output shape: {y.shape}")
    
    # 6. Manual Verification
    # Emulate the logic
    w_fp16 = weight_int8.to(torch.float16)
    w_flat = w_fp16.flatten()
    
    target_len = w_flat.numel()
    pad_len = (32 - (target_len % 32)) % 32
    if pad_len > 0:
        w_flat = F.pad(w_flat, (0, pad_len))
        
    w_reshaped = w_flat.view(-1, 32)
    scale_reshaped = weight_scale.to(torch.float16).view(-1, 1)
    
    w_dequant = w_reshaped * scale_reshaped
    w_dequant_flat = w_dequant.flatten()
    
    if pad_len > 0:
        w_dequant_flat = w_dequant_flat[:-pad_len]
        
    w_ref = w_dequant_flat.view(out_channels, in_channels, kernel_size, kernel_size)
    
    b_ref = block.bias
    
    y_ref = F.conv2d(x, w_ref, b_ref, stride=1, padding=0, dilation=1, groups=1)
    
    diff = (y - y_ref).abs().max()
    print(f"Max difference: {diff}")
    
    if diff < 1e-3:
        print("✅ Test Passed!")
    else:
        print("❌ Test Failed!")

if __name__ == "__main__":
    test_conv2d_block_quant()
