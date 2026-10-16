# v0.1.0 安装与使用说明

作者：yangdewei；联系邮箱：1147398767@qq.com。

## 功能边界

这是独立的 FC3 傅里叶变换和本征模态缩并内核，支持 NumPy 与 CuPy，
采用 float64/complex128。不是完整 WTE 求解器，也不是给生产环境自动打补丁的插件。
不附带自定义 Wigner 插件、材料 FC、势函数或计算账号。

仅支持已验证的 Phono3py 3.31.1 接口、Phonopy 3.5.x、原生 r0-average=True、
symmetrize_fc3q=False。Phono3py 4.x 不受支持。
CPU 无 CUDA 也能使用 NumPy 后端；GPU 后端需 NVIDIA CUDA/CuPy 可用环境。

## 独立安装

建议 Linux + NVIDIA GPU，创建全新环境，不在生产环境执行安装。
支持 Python >=3.10；本轮 Linux GPU 实测使用 Python 3.10.21。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 在 wheel 所在目录执行；CUDA 12 环境：
python -m pip install './phono3py_cupy_accelerator-0.1.0-py3-none-any.whl[phono3py,cuda12]'
```

只用 CPU 时将 extras 改为 `[phono3py]`，无需安装 CuPy。
CUDA 版本应与安装的 CuPy 发行包及驱动兼容，不能在生产环境盲目升级。
安装前核对同目录 `SHA256SUMS.txt`；Linux 可运行 `sha256sum -c SHA256SUMS.txt`。

## 运行公开示例和测试

解压完整源码 ZIP 后，在源码根目录、已激活的新环境中执行：

```bash
python -m pip install 'pytest>=8,<10'
python examples/minimal.py --backend numpy
# 物理 GPU 序号或 UUID 由使用者显式选择；0 仅为示例，不代表空闲。
CUDA_VISIBLE_DEVICES=0 python examples/minimal.py --backend cupy
python -m pytest -q
ACCELERATOR_TEST_GPU=1 CUDA_VISIBLE_DEVICES=0 python -m pytest -q
```

最后一条开启全部 GPU 测试；无 CUDA 时不能把跳过的测试算作 PASS。
示例使用公开合成数据，不输出真实材料热导率。共享 GPU 需先得到资源所有者同意，
检查显存余量；本次共享测试不构成性能承诺。

## 接入自己的 FC 和声子

本软件的输入接口是已构造的 Phono3py `Interaction`，不是一个完整 WTE 配置文件。
结构、FC2、FC3、超胞/原胞矩阵、质量、网格和频率截断应由调用方通过
Phono3py 3.31.1 构造和核对；解好所需 q 点的频率/本征矢后，传入 `pp`：

```python
from phono3py_cupy_accelerator import FourierBatch
from phono3py_cupy_accelerator.phono3py_adapter import evaluate_triplet

# pp = 你已构造并完成谐波求解的 ph.phph_interaction
# pp.set_grid_point(...) 后获取原生、满足动量条件的 triplet：
triplet = pp.get_triplets_at_q()[0][0]
cpu = FourierBatch(pp, 'numpy')
v_numpy, time_numpy = evaluate_triplet(
    pp, triplet, backend='numpy', fourier=cpu)
gpu = FourierBatch(pp, 'cupy')
v_cupy, time_cupy = evaluate_triplet(
    pp, triplet, backend='cupy', fourier=gpu)
```

返回值是该 triplet 的完整 `|V3|²`，不是 gamma 或 kappa。
温度不会传给傅里叶/缩并内核，因为这两阶段不含 Bose 占据数。
温度、积分权重、triplet 权重、g_zero、gamma 求和、简并处理和 WTE 速度矩阵
仍由下游已验证程序处理，不能宣称修改一个温度参数即可由本包计算热导率。

FC3 可以是完整或紧凑数组，但必须与结构和映射完全匹配。
不得只删去吸附原子而继续使用原 FC。修改 FC、网格或映射后要重新建立 FourierBatch，
禁止沿用旧缓存。默认打包常量预算 2 GiB，不是 GPU 总显存上限；稠密张量仍随分支数立方增长。
当前缩并采用完整首级 GEMM，`block` 参数仅保留兼容性，不提供显存不足时的自动分块保证。

## 查看结果和复现性能

示例输出 JSON 数值误差与计时；内核 API 返回值和计时字典由调用方保存。
`benchmarks/benchmark_fourier.py` 与 `benchmark_contraction.py` 提供合成基准。
计时应区分首次初始化、常量上传、稳态内核、传输和完整墙钟；共享卡结果不能冒充独占 A/B。
本次打包测试不重复材料 M14，历史全流程性能与精度范围见英文 README。

安装包、完整源码、构建记录与独立的 `VALIDATION.json` 一起交付。
保持 BSD-3-Clause 及上游署名；本地交付不等于已上传 GitHub/PyPI，也不等于生产迁移批准。
