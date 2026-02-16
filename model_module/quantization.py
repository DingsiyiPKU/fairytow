import torch
import torch.nn as nn
import math

from dataclasses import dataclass

@dataclass
class QuantizationConfig:
    # Quant 参数
    clamp_min: float = 1e-6
    mu: float = 0.4 # 


@torch.no_grad()
def quantize_complex_tensor(w_real: torch.Tensor, w_imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply PhaseQuant logic to complex weight tensors"""
    phase = torch.angle(w_real + 1j * w_imag)

    real_pos = (phase >= -math.pi / 4) & (phase < math.pi / 4)
    real_neg = (phase >= 3 * math.pi / 4) | (phase < -3 * math.pi / 4)
    imag_pos = (phase >= math.pi / 4) & (phase < 3 * math.pi / 4)
    imag_neg = (phase >= -3 * math.pi / 4) & (phase < -math.pi / 4)

    mask_real = real_pos | real_neg
    mask_imag = imag_pos | imag_neg

    s_re = w_real[mask_real].abs().mean() if mask_real.any() else torch.tensor(0.0, device=w_real.device)
    s_im = w_imag[mask_imag].abs().mean() if mask_imag.any() else torch.tensor(0.0, device=w_imag.device)
    
    s_re = torch.clamp(s_re, min=1e-6)
    s_im = torch.clamp(s_im, min=1e-6)
    if torch.isnan(s_re) or torch.isinf(s_re): s_re = torch.tensor(1e-6, device=w_real.device)
    if torch.isnan(s_im) or torch.isinf(s_im): s_im = torch.tensor(1e-6, device=w_imag.device)

    qw_real = torch.zeros_like(w_real)
    qw_imag = torch.zeros_like(w_imag)
    
    qw_real[real_pos] = 1.0
    qw_real[real_neg] = -1.0
    qw_imag[imag_pos] = 1.0
    qw_imag[imag_neg] = -1.0

    qw_real_scaled = qw_real * s_re
    qw_imag_scaled = qw_imag * s_im
    return qw_real_scaled.to(w_real.dtype), qw_imag_scaled.to(w_imag.dtype)

@torch.no_grad()
def quantize_complex_tensor_fairy2w(w_real: torch.Tensor, w_imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply PhaseQuant logic to complex weight tensors"""
    config = QuantizationConfig()

    argument = torch.angle(w_real + 1j * w_imag)
    norm = torch.abs(w_real + 1j * w_imag)
    mean_norm = norm.mean()
    threshold = config.mu * mean_norm

    nonzeros = (norm >= threshold)
    # zeros = (norm < threshold)
    phase_1 = (argument > -math.pi / 3) & (argument <= math.pi / 3) & nonzeros          # 中心在 0°
    phase_2 = (argument > math.pi / 3) & (argument <= math.pi) & nonzeros               # 中心在 120°
    phase_3 = (argument <= -math.pi / 3) & nonzeros               # 中心在 240°



#    s_phase_1 = norm[phase_1].mean() if phase_1.any() else torch.tensor(0.0, device=w_real.device)
#    s_phase_2 = norm[phase_2].mean() if phase_2.any() else torch.tensor(0.0, device=w_real.device)
#    s_phase_3 = norm[phase_3].mean() if phase_3.any() else torch.tensor(0.0, device=w_real.device)


    SIN60 = 0.8660254037844386  # sqrt(3)/2

    # --- 计算三个投影分量 ---
    # P1 就是 w_real，不需要额外计算
    # P2 = -0.5 * real + 0.866 * imag
    # P3 = -0.5 * real - 0.866 * imag
    p2 = -0.5 * w_real + SIN60 * w_imag
    p3 = -0.5 * w_real - SIN60 * w_imag

    # --- 统计各个 Phase 的平均投影值 ---
    # 注意：每个 phase 使用其对应的投影变量
    s_phase_1 = w_real[phase_1].mean() if phase_1.any() else torch.tensor(0.0, device=w_real.device)
    s_phase_2 = p2[phase_2].mean() if phase_2.any() else torch.tensor(0.0, device=w_real.device)
    s_phase_3 = p3[phase_3].mean() if phase_3.any() else torch.tensor(0.0, device=w_real.device)
    
    s_phase_1 = torch.clamp(s_phase_1, min=config.clamp_min)
    s_phase_2 = torch.clamp(s_phase_2, min=config.clamp_min)
    s_phase_3 = torch.clamp(s_phase_3, min=config.clamp_min)
    if torch.isnan(s_phase_1) or torch.isinf(s_phase_1):
        s_phase_1 = torch.tensor(config.clamp_min, device=w_real.device)
    if torch.isnan(s_phase_2) or torch.isinf(s_phase_2):
        s_phase_2 = torch.tensor(config.clamp_min, device=w_real.device)
    if torch.isnan(s_phase_3) or torch.isinf(s_phase_3):
        s_phase_3 = torch.tensor(config.clamp_min, device=w_real.device)

    qw_real = torch.zeros_like(w_real)
    qw_imag = torch.zeros_like(w_imag)

    tmp = math.sqrt(3) / 2.0
    qw_real[phase_1] += 1.0 * s_phase_1
    qw_real[phase_2] += -0.5 * s_phase_2
    qw_imag[phase_2] += tmp * s_phase_2
    qw_real[phase_3] += -0.5 * s_phase_3
    qw_imag[phase_3] += -tmp * s_phase_3

    return qw_real.to(w_real.dtype), qw_imag.to(w_imag.dtype)

def apply_complex_inspired_quantization(model: nn.Module):
    """Apply complex-inspired quantization to real-valued model"""
    print("Applying complex-inspired quantization (PhaseQuant-based)...")
    
    @torch.no_grad()
    def quantize_linear_layer(module: nn.Linear):
        A = module.weight.data
        if A.shape[0] % 2 != 0 or A.shape[1] % 2 != 0:
            print(f"  -> Skipping layer (non-even dimensions): {A.shape}")
            return

        n, m = A.shape[0] // 2, A.shape[1] // 2
        A11, A12 = A[:n, :m], A[:n, m:]
        A21, A22 = A[n:, :m], A[n:, m:]

        U_re = 0.5 * (A11 + A22)
        U_im = 0.5 * (A21 - A12)
        W_re = 0.5 * (A11 - A22)
        W_im = 0.5 * (A12 + A21)

        U_re_q, U_im_q = quantize_complex_tensor(U_re, U_im)
        W_re_q, W_im_q = quantize_complex_tensor(W_re, W_im)

        A11_q = W_re_q + U_re_q
        A12_q = W_im_q - U_im_q
        A21_q = W_im_q + U_im_q
        A22_q = -W_re_q + U_re_q

        A_quant_top = torch.cat([A11_q, A12_q], dim=1)
        A_quant_bottom = torch.cat([A21_q, A22_q], dim=1)
        A_quant = torch.cat([A_quant_top, A_quant_bottom], dim=0)

        module.weight.data = A_quant.to(A.dtype)

    model.apply(lambda module: quantize_linear_layer(module) if isinstance(module, nn.Linear) else None)
    print("Complex-inspired quantization completed.")
    return model

def apply_fairy2w_quantization(model: nn.Module):
    """Apply complex-inspired quantization to real-valued model"""
    print("Applying fairy2w quantization (PhaseQuant-based)...")
    
    @torch.no_grad()
    def quantize_linear_layer(module: nn.Linear):
        A = module.weight.data
        if A.shape[0] % 2 != 0 or A.shape[1] % 2 != 0:
            print(f"  -> Skipping layer (non-even dimensions): {A.shape}")
            return

        n, m = A.shape[0] // 2, A.shape[1] // 2
        A11, A12 = A[:n, :m], A[:n, m:]
        A21, A22 = A[n:, :m], A[n:, m:]

        U_re = 0.5 * (A11 + A22)
        U_im = 0.5 * (A21 - A12)
        W_re = 0.5 * (A11 - A22)
        W_im = 0.5 * (A12 + A21)

        U_re_q, U_im_q = quantize_complex_tensor_fairy2w(U_re, U_im)
        W_re_q, W_im_q = quantize_complex_tensor_fairy2w(W_re, W_im)

        A11_q = W_re_q + U_re_q
        A12_q = W_im_q - U_im_q
        A21_q = W_im_q + U_im_q
        A22_q = -W_re_q + U_re_q

        A_quant_top = torch.cat([A11_q, A12_q], dim=1)
        A_quant_bottom = torch.cat([A21_q, A22_q], dim=1)
        A_quant = torch.cat([A_quant_top, A_quant_bottom], dim=0)

        module.weight.data = A_quant.to(A.dtype)

    model.apply(lambda module: quantize_linear_layer(module) if isinstance(module, nn.Linear) else None)
    print("Complex-inspired quantization completed.")
    return model

def apply_bitnet_quantization(model: nn.Module):
    """Apply BitNet 1-bit quantization to real-valued model"""
    print("Applying BitNet (true 1-bit, affine) quantization to real-valued model...")
    
    @torch.no_grad()
    def quantize_linear_layer(module: nn.Linear):
        scale = module.weight.data.abs().mean()
        alpha = module.weight.data.mean()
        centered_weights = module.weight.data - alpha
        binarized_weights = torch.where(centered_weights > 0, 1.0, -1.0)
        module.weight.data = binarized_weights.to(module.weight.data.dtype) * scale

    model.apply(lambda module: quantize_linear_layer(module) if isinstance(module, nn.Linear) else None)
    print("BitNet quantization completed.")
    return model

def apply_bitnet_1_58bit_quantization_standard(model: nn.Module):
    """Apply BitNet 1.58-bit quantization to real-valued model (quantize to {-1, 0, +1})"""
    print("Applying BitNet 1.58-bit (absmean threshold) quantization to real-valued model...")
    
    @torch.no_grad()
    def quantize_linear_layer(module: nn.Linear):
        W = module.weight.data
        gamma = W.abs().mean()
        W_normalized = W / (gamma + 1e-5)
        W_quantized = torch.clamp(torch.round(W_normalized), -1.0, 1.0)
        module.weight.data = W_quantized.to(W.dtype) * gamma
    
    model.apply(lambda module: quantize_linear_layer(module) if isinstance(module, nn.Linear) else None)
    print("BitNet 1.58-bit (absmean threshold) quantization completed.")
    return model

def apply_bitnet_1_58bit_quantization_variant(model: nn.Module, threshold: float = 0.5):
    """Apply BitNet 1.58-bit quantization to real-valued model (quantize to {-1, 0, +1})"""
    print("Applying BitNet 1.58-bit (ternary) quantization to real-valued model...")
    
    @torch.no_grad()
    def quantize_linear_layer(module: nn.Linear):
        gamma = module.weight.data.abs().mean()
        normalized_weights = module.weight.data / (gamma + 1e-5)
        adaptive_threshold = threshold
        ternary_weights = torch.zeros_like(normalized_weights)
        ternary_weights[normalized_weights > adaptive_threshold] = 1.0
        ternary_weights[normalized_weights < -adaptive_threshold] = -1.0
        module.weight.data = ternary_weights.to(module.weight.data.dtype) * gamma
    
    model.apply(lambda module: quantize_linear_layer(module) if isinstance(module, nn.Linear) else None)
    print("BitNet 1.58-bit quantization completed.")
    return model

def minmax_1bit_quantize_dequantize(w: torch.Tensor) -> torch.Tensor:
    """Apply 1-bit Min-Max quantization and dequantization to weight tensor"""
    min_val = w.min()
    max_val = w.max()
    scale = (max_val - min_val) / 1.0
    zero_point = min_val

    if abs(scale) < 1e-9:
        return w

    quantized_w = torch.round((w - zero_point) / scale)
    dequantized_w = quantized_w * scale + zero_point
    
    return dequantized_w.to(w.dtype)

def apply_minmax_1bit_quantization(model: nn.Module):
    """Apply Min-Max 1-bit quantization to real-valued model"""
    print("Applying Min-Max (1-bit) quantization to real-valued model...")
    
    @torch.no_grad()
    def quantize_linear_layer(module: nn.Linear):
        module.weight.data = minmax_1bit_quantize_dequantize(module.weight.data)

    model.apply(lambda module: quantize_linear_layer(module) if isinstance(module, nn.Linear) else None)
    
    print("Min-Max 1-bit quantization completed.")
    return model

def symmetric_minmax_1bit_quantize_dequantize(w: torch.Tensor) -> torch.Tensor:
    """Apply symmetric 1-bit Min-Max quantization to weight tensor (quantize to {-1, 1})"""
    max_abs = w.abs().max()
    scale = max_abs

    if scale < 1e-9:
        return w

    quantized_w = (w / scale).sign()
    dequantized_w = quantized_w * scale
    
    return dequantized_w.to(w.dtype)

def apply_symmetric_minmax_1bit_quantization(model: nn.Module):
    """Apply symmetric Min-Max 1-bit quantization to real-valued model"""
    print("Applying symmetric Min-Max (1-bit, to {-1, 1}) quantization to real-valued model...")
    
    @torch.no_grad()
    def quantize_linear_layer(module: nn.Linear):
        module.weight.data = symmetric_minmax_1bit_quantize_dequantize(module.weight.data)

    model.apply(lambda module: quantize_linear_layer(module) if isinstance(module, nn.Linear) else None)
    
    print("Symmetric Min-Max 1-bit quantization completed.")
    return model

class BitNetQuantSTE(torch.autograd.Function):
    """BitNet STE: quantize in forward, pass gradients in backward"""
    @staticmethod
    def forward(ctx, w):
        scale = w.abs().mean()
        alpha = w.mean()
        centered_w = w - alpha
        binarized_w = torch.where(centered_w > 0, 1.0, -1.0).to(w.dtype)
        quantized_w = binarized_w * scale
        return quantized_w

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BitNet1_58QuantSTE(torch.autograd.Function):
    """BitNet 1.58-bit STE: quantize to {-1, 0, +1}, pass gradients in backward"""
    @staticmethod
    def forward(ctx, w):
        gamma = w.abs().mean()
        w_normalized = w / (gamma + 1e-5)
        w_quantized = torch.clamp(torch.round(w_normalized), -1.0, 1.0)
        quantized_w = (w_quantized * gamma).to(w.dtype)
        return quantized_w
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class PhaseQuantSTE(torch.autograd.Function):
    """Complex-Phase STE: quantize in forward, pass gradients in backward"""
    @staticmethod
    def forward(ctx, w_real, w_imag):
        phase = torch.angle(w_real + 1j * w_imag)
        
        real_pos = (phase >= -math.pi / 4) & (phase < math.pi / 4)
        real_neg = (phase >= 3 * math.pi / 4) | (phase < -3 * math.pi / 4)
        imag_pos = (phase >= math.pi / 4) & (phase < 3 * math.pi / 4)
        imag_neg = (phase >= -3 * math.pi / 4) & (phase < -math.pi / 4)

        mask_real = real_pos | real_neg
        mask_imag = imag_pos | imag_neg
        
        s_re = w_real[mask_real].abs().mean() if mask_real.any() else torch.tensor(0.0, device=w_real.device)
        s_im = w_imag[mask_imag].abs().mean() if mask_imag.any() else torch.tensor(0.0, device=w_imag.device)
        
        s_re = torch.clamp(s_re, min=1e-6)
        s_im = torch.clamp(s_im, min=1e-6)
        
        qw_real = torch.zeros_like(w_real)
        qw_imag = torch.zeros_like(w_imag)
        
        qw_real[real_pos] = 1.0
        qw_real[real_neg] = -1.0
        qw_imag[imag_pos] = 1.0
        qw_imag[imag_neg] = -1.0
        
        qw_real_scaled = qw_real * s_re
        qw_imag_scaled = qw_imag * s_im
        
        return qw_real_scaled.to(w_real.dtype), qw_imag_scaled.to(w_imag.dtype)

    @staticmethod
    def backward(ctx, grad_w_real, grad_w_imag):
        return grad_w_real, grad_w_imag

class PhaseQuantSTE_V2(torch.autograd.Function):
    """Two-step residual quantization"""

    @staticmethod
    def forward(ctx, w_real: torch.Tensor, w_imag: torch.Tensor):
        qw_real_o1, qw_imag_o1 = PhaseQuantSTE.apply(w_real, w_imag)
        error_real = w_real - qw_real_o1
        error_imag = w_imag - qw_imag_o1
        qw_real_o2, qw_imag_o2 = PhaseQuantSTE.apply(error_real, error_imag)
        qw_real = qw_real_o1 + qw_real_o2
        qw_imag = qw_imag_o1 + qw_imag_o2
        return qw_real, qw_imag

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        return grad_real, grad_imag

class PhaseQuantSTE_V3(torch.autograd.Function):
    """Three-step residual quantization"""

    @staticmethod
    def forward(ctx, w_real: torch.Tensor, w_imag: torch.Tensor):
        qw_real_o1, qw_imag_o1 = PhaseQuantSTE.apply(w_real, w_imag)
        error_real_1 = w_real - qw_real_o1
        error_imag_1 = w_imag - qw_imag_o1
        qw_real_o2, qw_imag_o2 = PhaseQuantSTE.apply(error_real_1, error_imag_1)
        error_real_2 = error_real_1 - qw_real_o2
        error_imag_2 = error_imag_1 - qw_imag_o2
        qw_real_o3, qw_imag_o3 = PhaseQuantSTE.apply(error_real_2, error_imag_2)
        qw_real = qw_real_o1 + qw_real_o2 + qw_real_o3
        qw_imag = qw_imag_o1 + qw_imag_o2 + qw_imag_o3
        return qw_real, qw_imag

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        return grad_real, grad_imag

class PhaseQuantSTE_V4(torch.autograd.Function):
    """Four-step residual quantization"""

    @staticmethod
    def forward(ctx, w_real: torch.Tensor, w_imag: torch.Tensor):
        qw_real_o1, qw_imag_o1 = PhaseQuantSTE.apply(w_real, w_imag)
        error_real_1 = w_real - qw_real_o1
        error_imag_1 = w_imag - qw_imag_o1
        qw_real_o2, qw_imag_o2 = PhaseQuantSTE.apply(error_real_1, error_imag_1)
        error_real_2 = error_real_1 - qw_real_o2
        error_imag_2 = error_imag_1 - qw_imag_o2
        qw_real_o3, qw_imag_o3 = PhaseQuantSTE.apply(error_real_2, error_imag_2)
        error_real_3 = error_real_2 - qw_real_o3
        error_imag_3 = error_imag_2 - qw_imag_o3
        qw_real_o4, qw_imag_o4 = PhaseQuantSTE.apply(error_real_3, error_imag_3)
        qw_real = qw_real_o1 + qw_real_o2 + qw_real_o3 + qw_real_o4
        qw_imag = qw_imag_o1 + qw_imag_o2 + qw_imag_o3 + qw_imag_o4
        return qw_real, qw_imag

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        return grad_real, grad_imag

class Fairy2w_PhaseQuantSTE(torch.autograd.Function):
    """Fairy2w-Phase STE: quantize in forward, pass gradients in backward"""
    @staticmethod
    def forward(ctx, a, b):
        config = QuantizationConfig()
        # 1. 计算三个方向的投影 (Projection)
        p1 = a - 0.5 * b
        p2 = b - 0.5 * a
        p3 = -0.5 * (a + b)

        # 2. 计算全局阈值 (基于最大投影的平均值)
        # p_max 代表点在最亲和方向上的强度
        p_max = torch.max(p1, torch.max(p2, p3))
        threshold = config.mu * p_max.mean()

        # 3. 生成互斥掩码 (Priority: 1 > 2 > 3)
        # 这种逻辑确保了在边界线上（Tie-break）点只会被分配给一个区域
        nonzero = (p_max >= threshold)
        
        m1 = (p1 == p_max) & nonzero
        m2 = (p2 == p_max) & nonzero & (~m1)
        # 如果是非零且不属于区域1和2，则必然属于区域3
        m3 = nonzero & (~m1) & (~m2)

        
        '''import random
        a = random.random()  # 生成一个随机数，确保代码不会被优化掉
        if a < 0.2:  # 1% 的概率进入调试输出
            num_elements = a.numel()
            non_zero = nonzero.sum().item()
            pha_1 = m1.sum().item()
            pha_2 = m2.sum().item()
            pha_3 = m3.sum().item()
            # 1. 使用最原始的 print，并强制刷新缓冲区
            print(f"\n\n{'='*20} DEBUG OUTPUT {'='*20}", flush=True)
            print(f"Total elements: {num_elements}", flush=True)
            print(f"Non-zero elements: {non_zero}", flush=True)
            print(f"Phase 1 count: {pha_1/ num_elements:.2%}", flush=True)
            print(f"Phase 2 count: {pha_2/ num_elements:.2%}", flush=True)
            print(f"Phase 3 count: {pha_3/ num_elements:.2%}", flush=True)
            print(f"Sparsity: {(num_elements - non_zero) / num_elements:.2%}", flush=True)
            print(f"{'='*54}\n\n", flush=True)
            
            # 2. 不要用 sys.exit(0)，改用 raise RuntimeError
            # 这样能看到完整的调用栈，确保程序是因为运行到这里而停止的
            raise RuntimeError("DEBUG_STOP: Reach quantization forward")'''



        # 4. 计算各区域缩放因子 (Safe Mean 避免同步)
        def get_scale(p_vals, mask):
            # sum / count 逻辑
            sum_val = torch.sum(p_vals * mask)
            count = torch.sum(mask)
            # 使用 1e-8 避免除零，结果进行 clamp
            scale = sum_val / (count + 1e-8)
            return torch.clamp(scale, min=config.clamp_min)

        s1 = get_scale(p1, m1)
        s2 = get_scale(p2, m2)
        s3 = get_scale(p3, m3)

        # 5. 重建量化后的 qa, qb
        # 基底表示: 
        # m1 (1):   a=s1, b=0
        # m2 (w):   a=0,  b=s2
        # m3 (w^2): a=-s3, b=-s3 (根据 w^2 = -1 - w)
        
        m1_f = m1.float()
        m2_f = m2.float()
        m3_f = m3.float()

        qa = (m1_f * s1) + (m3_f * -s3)
        qb = (m2_f * s2) + (m3_f * -s3)

        return qa.to(a.dtype), qb.to(b.dtype)

    @staticmethod
    def backward(ctx, grad_qa, grad_qb):
        # STE (Straight-Through Estimator)
        return grad_qa, grad_qb


class Fairy2w_PhaseQuantSTE_Alter(torch.autograd.Function):
    """Use 60°, 180°, 300°"""
    @staticmethod
    def forward(ctx, a, b):
        config = QuantizationConfig()
        # 1. 计算三个方向的投影 (Eisenstein 线性组合)
        # 相比 real/imag 坐标系，这里省去了 sqrt(3) 的乘法，仅需 0.5 倍率
        p1 = 0.5 * (a + b)  # 60° 方向
        p2 = -a + 0.5 * b   # 180° 方向
        p3 = 0.5 * a - b    # 300° 方向

        # 2. 计算全局阈值
        p_max = torch.max(p1, torch.max(p2, p3))
        threshold = config.mu * p_max.mean()

        # 3. 优先级互斥掩码 (Priority: 60° > 180° > 300°)
        nonzero = (p_max >= threshold)
        m1 = (p1 == p_max) & nonzero
        m2 = (p2 == p_max) & nonzero & (~m1)
        m3 = nonzero & (~m1) & (~m2)

        # 4. 计算缩放因子 (Safe Mean)
        def get_scale(p_vals, mask):
            sum_val = torch.sum(p_vals * mask)
            count = torch.sum(mask)
            if count == 0:
                return torch.tensor(0.0, device=p_vals.device, dtype=p_vals.dtype)
            else:
                return torch.clamp(sum_val / (count + 1e-8), min=config.clamp_min)

        s1 = get_scale(p1, m1)
        s2 = get_scale(p2, m2)
        s3 = get_scale(p3, m3)

        # 5. 重建 qa, qb
        # v1 (60°):  z = s1 * (1 + w)  => a=s1, b=s1
        # v2 (180°): z = s2 * (-1)     => a=-s2, b=0
        # v3 (300°): z = s3 * (-w)     => a=0, b=-s3
        
        m1_f, m2_f, m3_f = m1.float(), m2.float(), m3.float()

        qa = (m1_f * s1) + (m2_f * -s2)
        qb = (m1_f * s1) + (m3_f * -s3)

        return qa.to(a.dtype), qb.to(b.dtype)

    @staticmethod
    def backward(ctx, grad_qa, grad_qb):
        return grad_qa, grad_qb

class Fairy2w_PhaseQuantSTE_V2(torch.autograd.Function):
    """Two-step residual quantization"""

    @staticmethod
    def forward(ctx, w_real: torch.Tensor, w_imag: torch.Tensor):
        qw_real_o1, qw_imag_o1 = Fairy2w_PhaseQuantSTE.apply(w_real, w_imag)
        error_real = w_real - qw_real_o1
        error_imag = w_imag - qw_imag_o1
        qw_real_o2, qw_imag_o2 = Fairy2w_PhaseQuantSTE_Alter.apply(error_real, error_imag)
        qw_real = qw_real_o1 + qw_real_o2
        qw_imag = qw_imag_o1 + qw_imag_o2
        return qw_real.to(w_real.dtype), qw_imag.to(w_real.dtype)

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        return grad_real, grad_imag

class Fairy2w_PhaseQuantSTE_V3(torch.autograd.Function):
    """Three-step residual quantization"""

    @staticmethod
    def forward(ctx, w_real: torch.Tensor, w_imag: torch.Tensor):
        qw_real_o1, qw_imag_o1 = Fairy2w_PhaseQuantSTE.apply(w_real, w_imag)
        error_real_1 = w_real - qw_real_o1
        error_imag_1 = w_imag - qw_imag_o1
        qw_real_o2, qw_imag_o2 = Fairy2w_PhaseQuantSTE_Alter.apply(error_real_1, error_imag_1)
        error_real_2 = error_real_1 - qw_real_o2
        error_imag_2 = error_imag_1 - qw_imag_o2
        qw_real_o3, qw_imag_o3 = Fairy2w_PhaseQuantSTE.apply(error_real_2, error_imag_2)
        qw_real = qw_real_o1 + qw_real_o2 + qw_real_o3
        qw_imag = qw_imag_o1 + qw_imag_o2 + qw_imag_o3
        return qw_real, qw_imag

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        return grad_real, grad_imag

class Fairy2w_PhaseQuantSTE_V4(torch.autograd.Function):
    """Four-step residual quantization"""

    @staticmethod
    def forward(ctx, w_real: torch.Tensor, w_imag: torch.Tensor):
        qw_real_o1, qw_imag_o1 = Fairy2w_PhaseQuantSTE.apply(w_real, w_imag)
        error_real_1 = w_real - qw_real_o1
        error_imag_1 = w_imag - qw_imag_o1
        qw_real_o2, qw_imag_o2 = Fairy2w_PhaseQuantSTE_Alter.apply(error_real_1, error_imag_1)
        error_real_2 = error_real_1 - qw_real_o2
        error_imag_2 = error_imag_1 - qw_imag_o2
        qw_real_o3, qw_imag_o3 = Fairy2w_PhaseQuantSTE.apply(error_real_2, error_imag_2)
        error_real_3 = error_real_2 - qw_real_o3
        error_imag_3 = error_imag_2 - qw_imag_o3
        qw_real_o4, qw_imag_o4 = Fairy2w_PhaseQuantSTE_Alter.apply(error_real_3, error_imag_3)
        qw_real = qw_real_o1 + qw_real_o2 + qw_real_o3 + qw_real_o4
        qw_imag = qw_imag_o1 + qw_imag_o2 + qw_imag_o3 + qw_imag_o4
        return qw_real, qw_imag

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        return grad_real, grad_imag



class Fairy2w_PhaseQuantSTE_Eisenstein(torch.autograd.Function):
    """
    Fairy2w-Phase STE: 基于 Eisenstein 基底的不对称斜坐标分解量化
    输入: a, b (满足 z = a + b*omega)
    """
    @staticmethod
    def forward(ctx, a, b):
        config = QuantizationConfig()
        
        # 1. 区域判定 (扇区划分)
        m1 = (a >= b) & (a >= 0)
        m2 = (b > a)  & (b >= 0)
        m3 = (a < 0)  & (b < 0)

        # 2. 不对称分解 c (基于用户手绘图逻辑)
        c = torch.zeros_like(a)
        # 区域 1
        c = torch.where(m1, torch.where(b > 0, a - b, a), c)
        # 区域 2
        c = torch.where(m2, torch.where(a > 0, b - a, b), c)
        # 区域 3
        c = torch.where(m3, torch.where(b > a, -b, -a), c)

        # 3. 计算放缩因子 S (包含区域内所有数的贡献)
        # 这里的 mask 使用的是大区域 m1, m2, m3，而不是 nonzero 后的掩码
        def get_global_region_scale(c_vals, region_mask):
            # 分子：区域内所有 c 的和 (包含那些即将变 0 的数)
            sum_val = torch.sum(c_vals * region_mask)
            # 分母：区域内所有位置的总数
            count = torch.sum(region_mask)
            # 计算平均值并 clamp
            scale = sum_val / (count + 1e-8)
            return torch.clamp(scale, min=config.clamp_min)

        s1 = get_global_region_scale(c, m1)
        s2 = get_global_region_scale(c, m2)
        s3 = get_global_region_scale(c, m3)

        # 4. 阈值化与稀疏处理
        threshold = config.mu * c.mean()
        nonzero = (c >= threshold)
        
        # 只有通过阈值的点才会被赋予缩放后的单位向量
        active1 = m1 & nonzero
        active2 = m2 & nonzero
        active3 = m3 & nonzero

        # 5. 重建输出
        # 虽然 s1 是由所有 m1 算出的，但只乘在 active1 上
        m1_f, m2_f, m3_f = active1.float(), active2.float(), active3.float()
        
        qa = (m1_f * s1) + (m3_f * -s3)
        qb = (m2_f * s2) + (m3_f * -s3)

        return qa.to(a.dtype), qb.to(b.dtype)

    @staticmethod
    def backward(ctx, grad_qa, grad_qb):
        # Straight-Through Estimator (STE)
        # 直接传递梯度给 a 和 b
        return grad_qa, grad_qb
    
    

class Fairy2w_PhaseQuantSTE_Alter_Eisenstein(torch.autograd.Function):
    """Use 60°, 180°, 300°"""
    @staticmethod
    def forward(ctx, a, b):
        config = QuantizationConfig()
        # 1. 计算三个方向的投影 (Eisenstein 线性组合)
        # 相比 real/imag 坐标系，这里省去了 sqrt(3) 的乘法，仅需 0.5 倍率
        p1 = 0.5 * (a + b)  # 60° 方向
        p2 = -a + 0.5 * b   # 180° 方向
        p3 = 0.5 * a - b    # 300° 方向

        # 2. 计算全局阈值
        p_max = torch.max(p1, torch.max(p2, p3))
        threshold = config.mu * p_max.mean()

        # 3. 优先级互斥掩码 (Priority: 60° > 180° > 300°)
        nonzero = (p_max >= threshold)
        m1 = (p1 == p_max) 
        m2 = (p2 == p_max) & (~m1)
        m3 = (~m1) & (~m2)

        # 4. 计算缩放因子 (Safe Mean)
        def get_scale(p_vals, mask):
            sum_val = torch.sum(p_vals * mask)
            count = torch.sum(mask)
            return torch.clamp(sum_val / (count + 1e-8), min=config.clamp_min)

        s1 = get_scale(p1, m1)
        s2 = get_scale(p2, m2)
        s3 = get_scale(p3, m3)

        # 5. 重建 qa, qb
        # v1 (60°):  z = s1 * (1 + w)  => a=s1, b=s1
        # v2 (180°): z = s2 * (-1)     => a=-s2, b=0
        # v3 (300°): z = s3 * (-w)     => a=0, b=-s3
        
        active1 = m1 & nonzero
        active2 = m2 & nonzero
        active3 = m3 & nonzero

        # 5. 重建输出
        # 虽然 s1 是由所有 m1 算出的，但只乘在 active1 上
        m1_f, m2_f, m3_f = active1.float(), active2.float(), active3.float()

        qa = (m1_f * s1) + (m2_f * -s2)
        qb = (m1_f * s1) + (m3_f * -s3)

        return qa.to(a.dtype), qb.to(b.dtype)

    @staticmethod
    def backward(ctx, grad_qa, grad_qb):
        return grad_qa, grad_qb
    
    
    
class Fairy2w_PhaseQuantSTE_V2_Eisenstein(torch.autograd.Function):
    """Two-step residual quantization"""

    @staticmethod
    def forward(ctx, w_real: torch.Tensor, w_imag: torch.Tensor):
        qw_real_o1, qw_imag_o1 = Fairy2w_PhaseQuantSTE_Eisenstein.apply(w_real, w_imag)
        error_real = w_real - qw_real_o1
        error_imag = w_imag - qw_imag_o1
        qw_real_o2, qw_imag_o2 = Fairy2w_PhaseQuantSTE_Alter_Eisenstein.apply(error_real, error_imag)
        return qw_real_o1.to(w_real.dtype), qw_imag_o1.to(w_real.dtype), qw_real_o2.to(w_real.dtype), qw_imag_o2.to(w_real.dtype)

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        return grad_real, grad_imag