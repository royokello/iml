import torch
import torch.nn.functional as F

C_IN = 1280
C_OUT = 1280
HEIGHT = 64
WIDTH = 64


def quantize_per_tensor(tensor: torch.Tensor):
    max_abs = tensor.abs().amax()
    scale = torch.clamp(max_abs / 127.0, min=1e-8).to(torch.float32)
    quantized = torch.clamp((tensor / scale).round(), -128, 127).to(torch.int8)
    return quantized, scale


def quantize_per_channel(weight: torch.Tensor):
    max_abs = weight.abs().amax(dim=1)
    scale = torch.clamp(max_abs / 127.0, min=1e-8).to(torch.float16)
    quantized = torch.clamp((weight / scale[:, None]).round(), -128, 127).to(
        torch.int8
    )
    return quantized, scale


def run_conv_1x1_accuracy_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    x_fp16 = torch.randn(1, C_IN, HEIGHT, WIDTH, dtype=torch.float16, device=device)
    w_fp16 = torch.randn(C_OUT, C_IN, 1, 1, dtype=torch.float16, device=device)
    bias_fp16 = torch.randn(C_OUT, dtype=torch.float16, device=device)

    y_ref = F.conv2d(x_fp16, w_fp16, bias_fp16)

    x_fp32 = x_fp16.float()                     # [C_in, H, W]
    w_slice = w_fp16.view(C_OUT, C_IN).float()   # [C_out, C_in]
    bias_fp32 = bias_fp16.float()               # [C_out]

    # Quantize
    x_q, scale_x = quantize_per_tensor(x_fp32)
    w_q, scale_w = quantize_per_channel(w_slice)

    # Flatten X for matmul: [C_in, H*W]
    x_q_flat = x_q.view(C_IN, HEIGHT * WIDTH).float()        # [C_in, HW]

    # W_q: [C_out, C_in] → want [C_in, C_out] for right-multiply
    w_q_t = w_q.float().transpose(0, 1)                      # [C_in, C_out]

    # Integer-like convolution (Z matrix)
    # Z = X_q_flat^T @ W_q    → [HW, C_out]
    z_int_like = (x_q_flat.transpose(0, 1) @ w_q_t)          # [HW, C_out]

    # Build combined scale
    scale_combined = (scale_x * scale_w).to(x_fp32.device)   # [C_out]

    print("1x1 conv accuracy test (per-tensor X, per-channel W)")
    # Apply Y = S * Z + B
    y_q = (z_int_like * scale_combined) + bias_fp32            # broadcasting: [HW, C_out]

    # Reshape back to [C_out, H, W]
    y_q = y_q.transpose(0, 1).view(C_OUT, HEIGHT, WIDTH)     # [C_out, H, W]

    diff = y_q - y_ref
    abs_diff = diff.abs()
    max_err = abs_diff.max().item()
    mean_err = abs_diff.mean().item()
    rmse = diff.pow(2).mean().sqrt().item()

    print(f"  max error : {max_err:.6e}")
    print(f"  mean error: {mean_err:.6e}")
    print(f"  rmse      : {rmse:.6e}")

    print("1x1 conv accuracy test (per-tensor X, per-tensor W)")
    _, scale_w = quantize_per_tensor(w_fp16)
    scale_combined = (scale_x * scale_w).to(x_fp32.device)
    y_q = (z_int_like * scale_combined) + bias_fp32
    y_q = y_q.transpose(0, 1).view(C_OUT, HEIGHT, WIDTH)

    diff = y_q - y_ref
    abs_diff = diff.abs()
    max_err = abs_diff.max().item()
    mean_err = abs_diff.mean().item()
    rmse = diff.pow(2).mean().sqrt().item()

    
    print(f"  max error : {max_err:.6e}")
    print(f"  mean error: {mean_err:.6e}")
    print(f"  rmse      : {rmse:.6e}")

    print("1x1 conv accuracy test (no scaling)")

    y_q = z_int_like + bias_fp32
    y_q = y_q.transpose(0, 1).view(C_OUT, HEIGHT, WIDTH)

    diff = y_q - y_ref
    abs_diff = diff.abs()
    max_err = abs_diff.max().item()
    mean_err = abs_diff.mean().item()
    rmse = diff.pow(2).mean().sqrt().item()

    
    print(f"  max error : {max_err:.6e}")
    print(f"  mean error: {mean_err:.6e}")
    print(f"  rmse      : {rmse:.6e}")

def main():
    run_conv_1x1_accuracy_test()


if __name__ == "__main__":
    main()
