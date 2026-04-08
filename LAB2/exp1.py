# 实现Collatz猜想

print("验证Collatz猜想")

n = int(input("请输入一个正整数 n: "))
if n <= 0:
    print("请输入正整数")
else:
    print(n, end=" ")
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        print(n, end=" ")
