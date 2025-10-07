# GPU 架构
![alt text](image.png)
和CPU不同，GPU用于高度并行处理数据。它由大量简单的核心组成，能够同时处理多个线程。

## GPU的工作流程：
![alt text](image-1.png)
X86 Host：运行CPU程序，控制计算流程，包括启动GPU任务和管理数据传输。

GPU Device：执行并行计算任务，包含多个计算单元（CU）和流处理器（SM）。

二者通过PCLe总线连接，并通过DMA直接访问彼此内存，无需CPU干预。

GPU工作流程：
1. 主机（CPU）将计算任务（内核函数）提交至 GPU 的硬件执行队列，并通过 PCIe 总线利用 DMA 将输入数据从主机内存传输到 GPU 的全局内存（Global Memory）。

2. GPU 的调度单元将任务划分为 Warp（32 线程一组），并分发给各个 Streaming Multiprocessor（SM）。每个 SM 的 Warp 调度器动态调度 Warp 执行，以隐藏内存延迟。

3. 当线程访问数据时，全局内存的请求经由 L2 Cache 和 L1 Cache（或 Texture Cache）加载到寄存器。若程序需要线程间协作，程序员可显式将数据从全局内存加载到 Shared Memory（片上高速存储），供同一 Block 内线程共享。

4. 计算结果最终写回全局内存，主机再通过 DMA 将结果传回主机内存。 

## SM 结构
![alt text](image-3.png)
  - 64 single precision cores（FP32）
  - 32 double precision cores（FP64）
  - 64 INT32 cores
  - 8 Tensor cores
  - 128KB memory_block for L1 cache and Shared memory  

    （default 0–96KB for Shared memory, 96–128KB for L1 cache）
    
    Shared memroy is **private** for block.

  - 65536 registers.

    Threads' function is to read the value of register memory which is obtained from global memory or shared memory implicitly. 

    所以如果在block里面的线程所访问的数据复用性不高，那么使用shared memory的意义并不是很大

Global_memory is large and shared by all SMs.

## Shared memory and bank
shared memory被组织成了多个可以并行访问的bank， 可理解为独立的存储单元， 每个bank可以独立地进行读写操作。

一般来说shared memoery = 96kB = 32 * 4B = 32 bank。 存储时0, 32, 64 ... 被映射到同一个bank。
### shared memory 读数
每个thread在shared memory中读数是自由的，可以读取任何位置， 例如thread0， 可以读取任意bank的任意行。 一个warp中的线程读数无限制， 可以读取相同的数据，也可以读取的数据均在同一个bank上。

### Bank conflict
一个warp中的线程发生了访问同一个bank不同数据的情况， 导致warp读取数据耗时延长。
### 广播
warp中的线程读取同一个数据， 对应bank只需读取一次数据后广播道所需的线程中。

## Shared memory usage
1. **静态申请**
```cpp
constexpr int W = 16;
constexpr int H = 16;
__global__ void g2s(int *origin){
  __shared__ int smem[H][W];
}
```
2. shared memory的数据是先让线程从global memory进行导入，读取完之后线程使用对应数据时就可以从shared memory中访问了
```cpp
int tx = threadIdx.x;
int ty = threadIdx.y;
smem[ty][tx] = origin[ty * W + tx];//orgin指向数据在global memory的位置
```
3. **shared memory的动态内存申请**
```cpp 
__global__ void g2s()
{
    extern __shared__ char shared_bytes[];
}
int main()
{
  //smem_size: 申请大小
    g2s<<<grid, block, smem_size>>>()
}
```

