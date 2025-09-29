import torch

def main():
    # 定义一个简单的函数：平方
    def square_fn(x):
        return torch.square(x)

    # 使用 torch.compile 编译
    compiled_square = torch.compile(square_fn)

    # 构造输入张量
    x = torch.randn(1024, device="cuda")

    # 执行一次，触发编译
    y = compiled_square(x)

    print("Input:", x[:5])
    print("Output:", y[:5])

if __name__ == "__main__":
    main()
