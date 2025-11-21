import torch
import torch.nn.functional as F

C_IN = 1280
C_OUT = 1280


def matmul_int8(w_q: torch.Tensor, x_q: torch.Tensor) -> torch.Tensor:
    # Torch's CUDA backend lacks AddMV kernels for int8/char, so run the
    # integer matmul on CPU when needed and always accumulate in int32.
    if w_q.device.type == "cuda":
        w_host = w_q.cpu().to(torch.int32)
        x_host = x_q.cpu().to(torch.int32)
        return torch.matmul(w_host, x_host).to(w_q.device)
    return torch.matmul(w_q.to(torch.int32), x_q.to(torch.int32))


def quantize_per_tensor(tensor: torch.Tensor):
    # Per-tensor int8 quantization
    # tensor: [...], scale: scalar
    max_abs = tensor.abs().amax()
    scale = torch.clamp(max_abs / 127.0, min=1e-8).float()
    quantized = torch.clamp((tensor / scale).round(), -128, 127).to(torch.int8)
    return quantized, scale


def quantize_per_channel(weight: torch.Tensor):
    # Per-output-channel int8 quantization
    # weight: [C_out, C_in], scale: [C_out]
    max_abs = weight.abs().amax(dim=1)
    scale = torch.clamp(max_abs / 127.0, min=1e-8).float()
    quantized = torch.clamp((weight / scale[:, None]).round(), -128, 127).to(torch.int8)
    return quantized, scale


def run_linear_accuracy_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # FP16 reference setup
    # x_fp16: [1, C_in]
    # w_fp16: [C_out, C_in]
    # b_fp16: [C_out]
    x_fp16 = torch.randn(1, C_IN, dtype=torch.float16, device=device)
    w_fp16 = torch.randn(C_OUT, C_IN, dtype=torch.float16, device=device)
    bias_fp16 = torch.randn(C_OUT, dtype=torch.float16, device=device)

    # Reference linear: Y_ref = X * W^T + b
    # y_ref_full: [1, C_out] → we remove batch: [C_out]
    y_ref = F.linear(x_fp16, w_fp16, bias_fp16)[0]

    # =====================================================================================
    # Test 1 — per-tensor activation scale (scalar Sx)
    #         per-channel weight scales (Sw[c], length C_out)
    # =====================================================================================

    # X quantization:
    # x_q: [C_in], scale_x: scalar
    x_q, scale_x = quantize_per_tensor(x_fp16[0])
    x_q = x_q.view(C_IN)

    # W quantization:
    # w_q: [C_out, C_in], scale_w: [C_out]
    w_q, scale_w = quantize_per_channel(w_fp16)

    # Integer-like linear:
    # Z[c] = sum_i X_q[i] * W_q[c,i]
    # z_int_like: [C_out]
    z_int_like = matmul_int8(w_q, x_q)

    # Combined scale per output channel:
    # Sc[c] = Sx * Sw[c]
    # scale_combined: [C_out]
    scale_combined = (scale_x * scale_w).to(device)

    # Y = Sc * Z + b
    # y_q: [C_out]
    y_q = (z_int_like * scale_combined) + bias_fp16

    # Accuracy stats
    diff = y_q - y_ref
    print("Linear accuracy test (per-tensor X, per-channel W)")
    print(f"  max error : {diff.abs().max().item():.6e}")
    print(f"  mean error: {diff.abs().mean().item():.6e}")
    print(f"  rmse      : {diff.pow(2).mean().sqrt().item():.6e}")

    # =====================================================================================
    # Test 2 — per-tensor activation scale
    #         per-tensor weight scale (scalar Sw)
    # =====================================================================================

    # W quantization (single scalar scale)
    w_q_pt, scale_w_pt = quantize_per_tensor(w_fp16)

    # Z_int_like using per-tensor W:
    z_int_like_pt = matmul_int8(w_q_pt, x_q)   # [C_out]

    # Combined scale is now scalar:
    scale_combined_pt = (scale_x * scale_w_pt).to(device)

    # y_q_pt: [C_out]
    y_q_pt = z_int_like_pt * scale_combined_pt + bias_fp16

    diff = y_q_pt - y_ref
    print("Linear accuracy test (per-tensor X, per-tensor W)")
    print(f"  max error : {diff.abs().max().item():.6e}")
    print(f"  mean error: {diff.abs().mean().item():.6e}")
    print(f"  rmse      : {diff.pow(2).mean().sqrt().item():.6e}")

    # =====================================================================================
    # Test 3 – No scaling
    # =====================================================================================

    # Equivalent to int32 accumulator + bias
    y_q_ns = z_int_like.to(torch.float32) + bias_fp16.to(torch.float32)     # [C_out]

    diff = y_q_ns - y_ref
    print("Linear accuracy test (no scaling)")
    print(f"  max error : {diff.abs().max().item():.6e}")
    print(f"  mean error: {diff.abs().mean().item():.6e}")
    print(f"  rmse      : {diff.pow(2).mean().sqrt().item():.6e}")


def main():
    run_linear_accuracy_test()


if __name__ == "__main__":
    main()
