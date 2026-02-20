
import triton
import triton.language as tl
import torch
from Fairy2w.model_module.quantization import Fairy2w_PhaseQuantSTE_V2_Eisenstein
import torch.nn.functional as F
def get_cuda_autotune_config():
    return [
        triton.Config(
            {'BLOCK_SIZE':M},num_stages=s, num_warps=w
        )
        for M in [32,64,128]
        for s in [1,2,3,4,5]
        for w in [4,8,16]
    ]
    



@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M','N'],
)  
@triton.jit
def fairytow_split_kernel(A_ptr,U_re_ptr,U_im_ptr,W_re_ptr,W_im_ptr,A_row_stride,A_col_stride,M, N,SM_NUM :  tl.constexpr,BLOCK_SIZE: tl.constexpr ):
# A是2M * 2N 大小的矩阵
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE)
    TOTAL_TASK = num_pid_m * num_pid_n    


    for pid in range(start_pid,TOTAL_TASK,SM_NUM):
        row_pid = pid // num_pid_n 
        col_pid = pid % num_pid_n
        
#A11, A12 = A[:n, :m], A[:n, m:]
#A21, A22 = A[n:, :m], A[n:, m:]

        A11_ptr = tl.make_block_ptr(A_ptr,shape=[2*M,2*N],
                                    strides=[A_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        A12_ptr = tl.make_block_ptr(A_ptr,shape=[2*M,2*N],
                                    strides=[A_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE+N],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        A21_ptr = tl.make_block_ptr(A_ptr,shape=[2*M,2*N],
                                    strides=[A_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE+M,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        A22_ptr = tl.make_block_ptr(A_ptr,shape=[2*M,2*N],
                                    strides=[A_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE+M,col_pid*BLOCK_SIZE+N],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        
        A11 = tl.load(A11_ptr)
        A12 = tl.load(A12_ptr)
        A21 = tl.load(A21_ptr)  
        A22 = tl.load(A22_ptr) 
        
        
        U_re = (A11 - A12 + A21 + 2 * A22) / 3
        U_im = (-A11 - 2 * A12 + 2 * A21 + A22) / 3
        W_re = (2 * A11 + A12 - A21 - 2 * A22) / 3
        W_im = (A11 + 2 * A12 + A21 - A22) / 3     

        U_row_stride = A_row_stride//2
        U1_ptr = tl.make_block_ptr(U_re_ptr,shape=[M,N],
                                    strides=[U_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        U2_ptr = tl.make_block_ptr(U_im_ptr,shape=[M,N],
                                    strides=[U_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        W1_ptr = tl.make_block_ptr(W_re_ptr,shape=[M,N],
                                    strides=[U_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        W2_ptr = tl.make_block_ptr(W_im_ptr,shape=[M,N],
                                    strides=[U_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
                        
        tl.store(U1_ptr,U_re.to(A11.dtype))
        tl.store(U2_ptr,U_im.to(A11.dtype))
        tl.store(W1_ptr,W_re.to(A11.dtype))
        tl.store(W2_ptr,W_im.to(A11.dtype))
        
        
def fairytow_split(A):
    M,N = A.shape
    M,N = M//2, N//2
    U_re = torch.empty((M,N),dtype=A.dtype,device=A.device)
    U_im = torch.empty((M,N),dtype=A.dtype,device=A.device)
    W_re = torch.empty((M,N),dtype=A.dtype,device=A.device)
    W_im = torch.empty((M,N),dtype=A.dtype,device=A.device)
    assert A.is_cuda, "Input matrix A must be on CUDA device"
    A_row_stride = A.stride(0)
    A_col_stride = A.stride(1)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = (NUM_SMS,)
    fairytow_split_kernel[grid](A,U_re,U_im,W_re,W_im,A_row_stride,A_col_stride,M,N,NUM_SMS)
    return U_re,U_im,W_re,W_im






@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M','N'],
)  
@triton.jit
def fairytow_combine_kernel(B_ptr,U_re_ptr,U_im_ptr,W_re_ptr,W_im_ptr,U_res_re_ptr,U_res_im_ptr,W_res_re_ptr,W_res_im_ptr,A_row_stride,A_col_stride,M, N,SM_NUM :  tl.constexpr,BLOCK_SIZE: tl.constexpr ):
# A是2M * 2N 大小的矩阵
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE)
    TOTAL_TASK = num_pid_m * num_pid_n    


    for pid in range(start_pid,TOTAL_TASK,SM_NUM):
        row_pid = pid // num_pid_n 
        col_pid = pid % num_pid_n
        
#A11, A12 = A[:n, :m], A[:n, m:]
#A21, A22 = A[n:, :m], A[n:, m:]
        U_row_stride = A_row_stride // 2
        U1_ptr = tl.make_block_ptr(U_re_ptr,shape=[M,N],
                                    strides=[U_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        U2_ptr = tl.make_block_ptr(U_im_ptr,shape=[M,N],
                                    strides=[U_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        W1_ptr = tl.make_block_ptr(W_re_ptr,shape=[M,N],
                                    strides=[U_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        W2_ptr = tl.make_block_ptr(W_im_ptr,shape=[M,N],
                                    strides=[U_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        res_U1_ptr = tl.make_block_ptr(U_res_re_ptr,shape=[M,N],
                                    strides=[U_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        res_U2_ptr = tl.make_block_ptr(U_res_im_ptr,shape=[M,N],
                                    strides=[U_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        res_W1_ptr = tl.make_block_ptr(W_res_re_ptr,shape=[M,N],
                                    strides=[U_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        res_W2_ptr = tl.make_block_ptr(W_res_im_ptr,shape=[M,N],
                                    strides=[U_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        U1 = tl.load(U1_ptr) + tl.load(res_U1_ptr)
        
        U2 = tl.load(U2_ptr) + tl.load(res_U2_ptr)
           
        W1 = tl.load(W1_ptr) + tl.load(res_W1_ptr)
        
        W2 = tl.load(W2_ptr) + tl.load(res_W2_ptr)
        
        A11_q = U1 + W1
        A12_q = -U2 - W1 + W2
        A21_q = W2 + U2
        A22_q = U1 - U2 - W1                      

        
        
        A11_ptr = tl.make_block_ptr(B_ptr,shape=[2*M,2*N],
                                    strides=[A_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        A12_ptr = tl.make_block_ptr(B_ptr,shape=[2*M,2*N],
                                    strides=[A_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE+N],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        A21_ptr = tl.make_block_ptr(B_ptr,shape=[2*M,2*N],
                                    strides=[A_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE+M,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        A22_ptr = tl.make_block_ptr(B_ptr,shape=[2*M,2*N],
                                    strides=[A_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE+M,col_pid*BLOCK_SIZE+N],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
                 
                 
        tl.store(A11_ptr,A11_q)
        tl.store(A12_ptr,A12_q)
        tl.store(A21_ptr,A21_q)
        tl.store(A22_ptr,A22_q)

def fairytow_combine(U_re,U_im,W_re,W_im,U_res_re,U_res_im,W_res_re,W_res_im):
    M,N = U_re.shape
    B = torch.empty((2*M,2*N),dtype=U_re.dtype,device=U_re.device)
    assert B.is_cuda, "Output matrix B must be on CUDA device"
    B_row_stride = B.stride(0)
    B_col_stride = B.stride(1)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = (NUM_SMS,)
    fairytow_combine_kernel[grid](B,U_re,U_im,W_re,W_im,U_res_re,U_res_im,W_res_re,W_res_im,B_row_stride,B_col_stride,M,N,NUM_SMS)
    return B



def QATLinearComplexPhaseV2_quantize(A:torch.Tensor):
        U_re,U_im,W_re,W_im = fairytow_split(A)
        U_re, U_im, U_res_re,U_res_im = Fairy2w_PhaseQuantSTE_V2_Eisenstein(U_re,U_im)
        W_re, W_im, W_res_re,W_res_im = Fairy2w_PhaseQuantSTE_V2_Eisenstein(W_re,W_im)
        B = fairytow_combine(U_re,U_im,W_re,W_im,U_res_re,U_res_im,W_res_re,W_res_im)
        return B




@torch.compile(mode="default")
def QATLinearComplexPhaseV2_forward(x:torch.Tensor,A:torch.Tensor,bias : torch.Tensor = None):
        B = (QATLinearComplexPhaseV2_quantize(A) - A).detach() + A
        return F.linear(x, B, bias) 









class fairytow_quant_V2(torch.autograd.Function):
    def forward(self, A):
        U_re,U_im,W_re,W_im = fairytow_split(A)
        U_re, U_im, U_res_re,U_res_im = Fairy2w_PhaseQuantSTE_V2_Eisenstein(U_re,U_im)
        W_re, W_im, W_res_re,W_res_im = Fairy2w_PhaseQuantSTE_V2_Eisenstein(W_re,W_im)
        B = fairytow_combine(U_re,U_im,W_re,W_im,U_res_re,U_res_im,W_res_re,W_res_im)
        return B
    
    def backward(self, grad_output):
        # 反向传播时，grad_output是B的梯度
        # 需要计算U_re,U_im,W_re,W_im的梯度，并返回与A相同形状的梯度
        # 这里我们暂时不实现反向传播，直接返回None
        return grad_output


def check(A):
        n, m = A.shape[0] // 2, A.shape[1] // 2

        # 1. 严格按照 PyTorch Linear 的权重分块
        # A11 对应 [0:n, 0:m], A12 对应 [0:n, m:2m] -> 作用于 x 产生输出的前半部分
        A11, A12 = A[:n, :m], A[:n, m:]
        A21, A22 = A[n:, :m], A[n:, m:]

        # 2. 计算 U, W 的分量 (系数基于前面的 omega 推导)
        U1 = (A11 - A12 + A21 + 2 * A22) / 3
        U2 = (-A11 - 2 * A12 + 2 * A21 + A22) / 3
        W1 = (2 * A11 + A12 - A21 - 2 * A22) / 3
        W2 = (A11 + 2 * A12 + A21 - A22) / 3

        U1, U2,U3,U4 = Fairy2w_PhaseQuantSTE_V2_Eisenstein(U1, U2)
        W1, W2, W3, W4 = Fairy2w_PhaseQuantSTE_V2_Eisenstein(W1, W2)
        U1 = U1 + U3
        U2 = U2 + U4
        W1 = W1 + W3
        W2 = W2 + W4
        # 3. 输入分割
        M_aa = U1 + W1
        M_ab = -U2 - W1 + W2
        M_ba = U2 + W2
        M_bb = U1 - U2 - W1

        # 2. 拼成大矩阵 (Stitch back to a big weight matrix)
        # 形状从 (n, m) 变为 (2n, 2m)
        # 第一行块: [M_aa, M_ab], 第二行块: [M_ba, M_bb]
        W_row1 = torch.cat([M_aa, M_ab], dim=1)
        W_row2 = torch.cat([M_ba, M_bb], dim=1)
        W_final = torch.cat([W_row1, W_row2], dim=0)
        return W_final
#python -m Fairy2w-W2.train.kernel 



    
    
DEVICE = "cuda"
configs = []
configs = [
    triton.testing.Benchmark(
        x_names=["M"],  
        x_vals=[256 * i for i in range(6, 33)],  
        line_arg="provider",  
        line_vals=["triton","complie","naive"],  
        line_names=["triton","complie","naive"],  
        styles=[("blue","-"),("green", "-"),("red", "-")], 
        ylabel='MS',  
        plot_name="fairytow_quant-performance-MS-complie", 
        args={},  
    )
]
@triton.testing.perf_report(configs)
def benchmark_MS(M, provider):
    N =  M
    A_real = torch.randn((M, N), device=DEVICE, dtype=torch.bfloat16)
    B_real = torch.randn((M, N), device=DEVICE, dtype=torch.bfloat16)
    quantizer = fairytow_quant_V2.apply

    quantiles = [0.5, 0.2, 0.8]
    ms2, min_ms, max_ms = triton.testing.do_bench(lambda: F.linear(A_real,B_real), quantiles=quantiles) 
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: quantizer(A_real), quantiles=quantiles) 
    if provider == 'naive':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: check(A_real), quantiles=quantiles) 
    if provider == 'complie':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: QATLinearComplexPhaseV2_forward(A_real,B_real), quantiles=quantiles)
        ms = ms - ms2
    perf = lambda ms: ms+ms2
    
    return perf(ms), perf(max_ms), perf(min_ms)











if __name__ == "__main__":
    A = torch.randn((4096,4096),dtype=torch.bfloat16,device="cuda")
    B = fairytow_quant_V2.apply(A)
    C = check(A)
    print((B-C).abs().max())
    print((B-C).abs().mean())
    
    QATLinearComplexPhaseV2_forward(A,B)
    print("Test passed!")
    benchmark_MS.run(save_path='/root/data',show_plots=True, print_data=True)




#python -m Fairy2w.model_module.kernel 