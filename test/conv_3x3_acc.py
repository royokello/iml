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
    scale = torch.clamp(max_abs / 127.0, min=1e-8).to(torch.float32)
    quantized = torch.clamp((weight / scale[:, None]).round(), -128, 127).to(
        torch.int8
    )
    return quantized, scale


def run_conv_3x3_accuracy_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    x_fp16 = torch.randn(1, C_IN, HEIGHT, WIDTH, dtype=torch.float16, device=device)
    w_fp16 = torch.randn(C_OUT, C_IN, 3, 3, dtype=torch.float16, device=device)
    bias_fp16 = torch.randn(C_OUT, dtype=torch.float16, device=device)

    y_ref = F.conv2d(x_fp16, w_fp16, bias_fp16, padding=1)

    x_fp32 = x_fp16.float()
    w_flat = w_fp16.view(C_OUT, -1).float()
    bias_fp32 = bias_fp16.float()

    x_q, scale_x = quantize_per_tensor(x_fp32)
    w_q_pc, scale_w_pc = quantize_per_channel(w_flat)

    x_q_f = x_q.float()
    x_unf = F.unfold(x_q_f, kernel_size=3, padding=1, stride=1)  # [1, C_in*9, H*W]
    x_cols = x_unf[0].transpose(0, 1)                             # [H*W, C_in*9]

    w_q_mat = w_q_pc.float()                                      # [C_out, C_in*9]
    z_int_like = x_cols @ w_q_mat.transpose(0, 1)                 # [H*W, C_out]

    print("3x3 conv accuracy test (per-tensor X, per-channel W)")
    scale_combined_pc = (scale_x * scale_w_pc).to(x_fp32.device)  # [C_out]
    y_q = z_int_like * scale_combined_pc + bias_fp32              # [H*W, C_out]
    y_q = y_q.transpose(0, 1).view(1, C_OUT, HEIGHT, WIDTH)

    diff = y_q - y_ref
    abs_diff = diff.abs()
    max_err = abs_diff.max().item()
    mean_err = abs_diff.mean().item()
    rmse = diff.pow(2).mean().sqrt().item()
    print(f"  max error : {max_err:.6e}")
    print(f"  mean error: {mean_err:.6e}")
    print(f"  rmse      : {rmse:.6e}")

    print("3x3 conv accuracy test (per-tensor X, per-tensor W)")
    _, scale_w_pt = quantize_per_tensor(w_fp16)
    scale_combined_pt = (scale_x * scale_w_pt).to(x_fp32.device)
    y_q2 = z_int_like * scale_combined_pt + bias_fp32
    y_q2 = y_q2.transpose(0, 1).view(1, C_OUT, HEIGHT, WIDTH)

    diff = y_q2 - y_ref
    abs_diff = diff.abs()
    max_err = abs_diff.max().item()
    mean_err = abs_diff.mean().item()
    rmse = diff.pow(2).mean().sqrt().item()
    print(f"  max error : {max_err:.6e}")
    print(f"  mean error: {mean_err:.6e}")
    print(f"  rmse      : {rmse:.6e}")

    print("3x3 conv accuracy test (no scaling)")
    y_q3 = z_int_like + bias_fp32
    y_q3 = y_q3.transpose(0, 1).view(1, C_OUT, HEIGHT, WIDTH)

    diff = y_q3 - y_ref
    abs_diff = diff.abs()
    max_err = abs_diff.max().item()
    mean_err = abs_diff.mean().item()
    rmse = diff.pow(2).mean().sqrt().item()
    print(f"  max error : {max_err:.6e}")
    print(f"  mean error: {mean_err:.6e}")
    print(f"  rmse      : {rmse:.6e}")


def main():
    run_conv_3x3_accuracy_test()


if __name__ == "__main__":
    main()
