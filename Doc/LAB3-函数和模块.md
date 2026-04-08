# 人工智能数学原理与算法

**实验内容：函数和模块**  
**日期：** 2025 年 3 月 16 日

---

## 目录
- [人工智能数学原理与算法](#人工智能数学原理与算法)
  - [目录](#目录)
  - [一、定义和调用函数](#一定义和调用函数)
  - [二、局部变量和全局变量](#二局部变量和全局变量)
  - [三、默认值形参和关键字实参](#三默认值形参和关键字实参)
  - [四、可变数量的实参](#四可变数量的实参)
  - [五、函数式编程](#五函数式编程)
  - [六、递归](#六递归)
    - [1. 计算阶乘](#1-计算阶乘)
    - [2. 计算最大公约数](#2-计算最大公约数)
    - [3. 计算字符串反转](#3-计算字符串反转)
    - [4. 实现快速排序](#4-实现快速排序)
  - [七、创建和使用模块](#七创建和使用模块)
  - [八、实验任务](#八实验任务)
    - [1. 二分查找](#1-二分查找)
    - [2. 有理数的四则运算](#2-有理数的四则运算)

---

## 一、定义和调用函数

函数是一组语句，可以根据输入参数计算输出结果。把需要多次运行的代码写成函数，可以实现代码的重复利用。以函数作为程序的组成单位使程序更易理解和维护。

**函数的定义**包括函数头和函数体两部分。
* **函数头**以关键字 `def` 开始，之后是空格和函数的名称。函数名称后面是一对圆括号，括号内可为空 (表示无形参) 或包含一些形参。形参表示函数的输入值。如果形参的数量超过一个，它们之间用逗号分隔。
* **函数体**由一条或多条语句构成，完成函数的功能，相对函数头需要有四个空格的缩进。

**调用函数**的语法是在函数的名称后面加上一对圆括号，括号内可为空 (表示无实参) 或包含一些实参。如果实参的数量超过一个，它们之间用逗号分隔。这些实参必须和函数的形参在数量上相同，并且在顺序上一一对应。

函数可以返回多个结果，这些结果之间用逗号分隔，构成一个**元组**。

**程序 4.1：定义和调用函数（最大公约数）**
```python
def gcd(a, b):
    """ 辗转相减法求两个正整数a和b的最大公约数 """
    while a != b:
        if a > b:
           a -= b;
        else:
           b -= a;
    return a

print(gcd(156, 732))  
print(gcd(1280, 800))  
```

**程序 4.2：返回多个结果**
```python
def max_min(a, b):
    """ 计算a和b的最大值和最小值 """
    if a > b:
        return a, b
    else:
        return b, a

print(max_min(156, 34))  
print(max_min(12, 800))  
```

---

## 二、局部变量和全局变量

函数的形参和在函数体内定义的变量称为**局部变量**。局部变量只能在函数体内访问，在函数运行结束时即被销毁。

在函数体外定义的变量称为**全局变量**。全局变量在任何函数中都可以被访问，除非某个函数中定义了同名的局部变量。如果需要在函数体中修改某个全局变量，需要用 `global` 声明它。

**程序 4.3：局部变量与全局变量同名**
```python
b = 10
c = 15
def f(a):
    b = 20  # 此处定义的是局部变量 b
    return a + b + c

print(f(5))  
print('b = %d' % b)  # 全局变量 b 仍为 10
```

**程序 4.4：使用 global 关键字修改全局变量**
```python
b = 10
c = 15
def f(a):
    global b
    b = 20  # 修改全局变量 b
    return a + b + c

print(f(5))  
print('b = %d' % b)  # 全局变量 b 变为 20
```

---

## 三、默认值形参和关键字实参

函数头可以给一个或多个形参赋予默认值，这些形参称为**默认值形参**。这些默认值形参的后面不能出现普通的形参。

在调用函数的语句中，可以在一个或多个实参的前面写上其对应的形参的名称。这些实参称为**关键字实参**。此时实参的顺序不必和函数头中的形参的顺序保持一致。

如果一个函数的返回值有多个而且数量未知，可以把所有需要返回的结果存储在一个容器 (例如列表) 中，最后返回整个容器。 

**程序 4.5：默认值形参与关键字实参示例**
```python
import math
def get_primes(lb, ub=100):
    """ 求解给定取值范围[lb, ub]内的所有质数 """
    primes = []
    if lb % 2 == 0: lb += 1
    if ub % 2 == 0: ub -= 1
    for i in range(lb, ub + 1, 2):
        isPrime = True
        for j in range(2, math.ceil(math.sqrt(i)) + 1):
            if i % j == 0:
                isPrime = False
                break
        if isPrime: primes.append(i)
    return primes

print(get_primes(40, 50))  
print(get_primes(120, 140))  
print(get_primes(80))  # ub 使用默认值 100
print(get_primes(ub=150, lb=136))  # 关键字实参，顺序可变
```

---

## 四、可变数量的实参

函数可以接受未知数量的位置实参和关键字实参。形参 `*args` 是一个元组，可接受未知数量的位置实参 (即普通实参)。形参 `**kwargs` 是一个字典，可接受未知数量的关键字实参。

**程序 4.6：**
```python
def fun(*args, **kwargs):
    print(type(args), type(kwargs))
    print('The positional arguments are', args)
    print('The keyword arguments are', kwargs)

fun(1, 2.3, 'a', True, u=6, x='Python', f=3.1415)
```

---

## 五、函数式编程

Python 语言支持函数式编程 (functional programming) 范式的基本方式是函数具有和其他类型 (如 int、float 等) 同样的性质：
* 被赋值给变量；
* 作为实参传给被调用函数的形参；
* 作为函数的返回值。

**程序 4.7：函数作为实参（在 IPython 交互中）**
```python
In[1]: animals = ["elephant", "tiger", "rabbit", "goat", "dog", "penguin"]
In[2]: sorted(animals)
Out[2]: ['dog', 'elephant', 'goat', 'penguin', 'rabbit', 'tiger']
In[3]: sorted(animals, key=len)
Out[3]: ['dog', 'goat', 'tiger', 'rabbit', 'penguin', 'elephant']
In[4]: sorted(animals, key=len, reverse=True)
Out[4]: ['elephant', 'penguin', 'rabbit', 'tiger', 'goat', 'dog']
In[5]: def m1(s): return ord(min(s))
In[6]: sorted(animals, key=m1)
Out[6]: ['elephant', 'rabbit', 'goat', 'dog', 'tiger', 'penguin']
In[7]: def m2(s): return ord(min(s)), len(s)
In[8]: sorted(animals, key=m2)
Out[8]: ['goat', 'rabbit', 'elephant', 'dog', 'tiger', 'penguin']
```

如果一个函数在定义以后只使用一次，并且函数体可以写成一个表达式，则可以使用 `lambda` 函数语法将其定义成一个**匿名函数**：`g = lambda 形参列表: 函数体表达式`。

**程序 4.8：Lambda 函数作为实参**
```python
def map_fs(f, s):
    for i in range(len(s)): s[i] = f(s[i])
    return s

a = [1, 3, 5, 7, 9]
print(map_fs(lambda x: x+1, a))  
print(map_fs(lambda x: x*x-1, a))  
```

**程序 4.9 & 4.10：函数作为返回值**
```python
def key_fun(n):
    def m1(s): return ord(min(s))
    def m2(s): return ord(min(s)), len(s)

    ms = [None, len, m1, m2]
    return ms[n]

animals = ["elephant", "tiger", "rabbit", "goat", "dog", "penguin"]
for i in range(4):
    print(sorted(animals, key=key_fun(i)))

# 输出结果 (程序 4.10)：
# ['dog', 'elephant', 'goat', 'penguin', 'rabbit', 'tiger']
# ['dog', 'goat', 'tiger', 'rabbit', 'penguin', 'elephant']
# ['elephant', 'rabbit', 'goat', 'dog', 'tiger', 'penguin']
# ['goat', 'rabbit', 'elephant', 'dog', 'tiger', 'penguin']
```

---

## 六、递归

递归就是一个函数调用自己。当要求解的问题满足以下三个条件时，递归是有效的解决方法：
1. 原问题可以分解为一个或多个结构类似但规模更小的子问题。
2. 当子问题的规模足够小时可以直接求解，称为**递归的终止条件**；否则可以继续对子问题递归求解。
3. 原问题的解可由子问题的解合并而成。

用递归方法解决问题的过程是基于对问题的分析提出递归公式。

### 1. 计算阶乘
阶乘的定义本身就是一个递归公式：
$$ n! = \begin{cases} 1 & \text{if } n = 1 \\ n * (n - 1)! & \text{if } n > 1 \end{cases} $$

**程序 4.11：阶乘递归实现**
```python
def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n - 1)

print(factorial(10))  
```

### 2. 计算最大公约数
设 $a$ 和 $b$ 表示两个正整数。若 $a > b$，则易证 $a$ 和 $b$ 的公约数集合等于 $a - b$ 和 $b$ 的公约数集合，因此 $a$ 和 $b$ 的最大公约数等于 $a - b$ 和 $b$ 的最大公约数。若 $a = b$，则 $a$ 和 $b$ 的最大公约数等于 $a$。由此可总结出递归公式如下：

$$ \text{gcd}(a, b) = \begin{cases} a & \text{if } a = b \\ \text{gcd}(a - b, b) & \text{if } a > b \\ \text{gcd}(a, b - a) & \text{if } a < b \end{cases} $$

递归函数都可以转换成与其等价的迭代形式（例如程序 4.1 中的 `gcd` 函数）。

**程序 4.12：最大公约数递归实现**
```python
def gcd(a, b):
    if a == b:
        return a
    elif a > b:
        return gcd(a-b, b)
    else:
        return gcd(a, b-a)

print(gcd(156, 732))  
```

### 3. 计算字符串反转
字符串反转就是将原字符串中的字符的先后次序反转，例如 "ABCDE" 反转以后得到 "EDCBA"。分析过程如下：
* "ABCDE" = "ABCD" + "E"
* "EDCBA" = "E" + "DCBA"
* "DCBA" = reverse("ABCD")
* "E" = reverse("E")

终止条件是：由单个字符构成的字符串的反转就是原字符串。递归公式如下：
$$ \text{reverse}(s) = \begin{cases} s & \text{if len}(s) = 1 \\ s[-1] + \text{reverse}(s[:-1]) & \text{if len}(s) > 1 \end{cases} $$

**程序 4.13：字符串反转递归实现**
```python
def reverse(s):
    if len(s) == 1:
        return s
    else:
        return s[-1] + reverse(s[:-1])

print(reverse("ABCDE"))  
```

### 4. 实现快速排序
以序列 `[3, 6, 2, 9, 7, 3, 1, 8]` 为例：
* 以第一个元素 `3` 为基准对其进行调整，把比 `3` 小的元素移动到左边，大的移动到右边。调整结果为 `[2, 1, 3, 3, 6, 9, 7, 8]`，可看成三个列表的连接: `[2, 1]`, `[3, 3]`, 和 `[6, 9, 7, 8]`。
* 排序完成的结果是 `[1, 2, 3, 3, 6, 7, 8, 9]`，也可看成连接: `[1, 2]`, `[3, 3]`, 和 `[6, 7, 8, 9]`。
* 对子问题 `[2, 1]` 和 `[6, 9, 7, 8]` 继续进行排序即可。

递归公式如下：
$$ \text{qsort}(s) = \begin{cases} s & \text{if len}(s) \le 1 \\ \text{qsort}(\{i \in s | i < s[0]\}) + \\ \{i \in s | i = s[0]\} + \text{qsort}(\{i \in s | i > s[0]\}) & \text{if len}(s) > 1 \end{cases} $$

**程序 4.14：快速排序递归实现**
```python
def qsort(s):
    if len(s) <= 1: return s
    s_less = []; s_greater = []; s_equal = []
    for k in s:
        if k < s[0]:
            s_less.append(k)
        elif k > s[0]:
            s_greater.append(k)
        else:
            s_equal.append(k)
    return qsort(s_less) + s_equal + qsort(s_greater)

print(qsort([3, 6, 2, 9, 7, 3, 1, 8])) 
```

---

## 七、创建和使用模块

模块是一个包含了若干函数和语句的文件，文件名是模块的名称加上 `.py` 后缀。一个模块实现了某类功能，是规模较大程序的组成单位，易于重复利用代码。

每个模块都有一个全局变量 `__name__`。模块的使用方式有两种：
1. **作为独立的程序运行**：此时变量 `__name__` 的值为 `'__main__'`。
2. **被其他程序导入**：此时变量 `__name__` 的值为模块的名称。

**程序 4.15：创建一个打印日历的模块 (`month_calendar.py`)**
```python
"""
Module for printing the monthly calendar for the year and
the month specified by the user.
...
"""
import sys, math

def is_leap(year): 
    return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0

# ... (为节省篇幅，部分辅助测试函数省略，逻辑为计算当月1号是星期几等) ...

def get_m01_in_week(year, month):
    n1 = get_0101_in_week(year)
    n2 = get_num_days_from_0101_to_m01(year, month)
    n = (n1 + n2) % 7
    return n

def print_header(year, month):
    print("%d  %d " % (year, month))
    print("---------------------------")
    print("Sun Mon Tue Wed Thu Fri Sat")

def print_body(year, month):
    n = get_m01_in_week(year, month)
    print(n * 4 * ' ', end='')
    for i in range(1, get_num_days_in_month(year, month) + 1):
        print('%-04d' % i, end='')
        if (i + n) % 7 == 0: print()

def print_monthly_calendar(year, month):
    print_header(year, month)
    print_body(year, month)

if __name__ == '__main__':  
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == '-h'):
        print(__doc__)  
    elif len(sys.argv) == 2 and sys.argv[1] == 'test':
        test_all_functions()  
    else:
        import argparse  
        parser = argparse.ArgumentParser()
        parser.add_argument('--year', type=int, default=2022)
        parser.add_argument('--month', type=int, default=1)
        args = parser.parse_args()
        print_monthly_calendar(args.year, args.month) 
```

**程序 4.16：模块作为独立的程序运行**
```text
In[1]: cd D:\Python\src
Out[1]: D:\Python\src
In[2]: run month_calendar.py --year 2022 --month 10
2022  10
---------------------------
Sun Mon Tue Wed Thu Fri Sat
                        1
2   3   4   5   6   7   8
9   10  11  12  13  14  15
16  17  18  19  20  21  22
23  24  25  26  27  28  29
30  31
```

**模块被其他程序导入：**
如果使用模块的程序和模块文件在同一个目录下时，使用 `import` 语句导入即可（程序 4.17）。如果不在同一个目录下，需要将模块所在目录插入到列表 `sys.path` 中（程序 4.18）。

**程序 4.17：同目录下导入**
```python
import month_calendar
y, m, d = 2022, 9, 18
n = (month_calendar.get_m01_in_week(y, m) + d - 1) % 7
dw = "Sun Mon Tue Wed Thu Fri Sat"
print(dw[4*n:4*n+4])  
```

**程序 4.18：不同目录下修改 sys.path 导入**
```text
In[1]: run ymd.py
Out[1]: ... ModuleNotFoundError: No module named 'month_calendar'
In[2]: import sys; sys.path.insert(0, 'D:\Python\src')
In[3]: run ymd.py
Sun
```

**Python 标准库的 calendar 模块：**
Python 标准库的 `calendar` 模块的 `TextCalendar` 类已提供了输出日历的功能。
**程序 4.20：**
```python
from calendar import TextCalendar
tc = TextCalendar()
print(tc.formatmonth(2022, 10))
print(tc.formatyear(2022, m=4))
```

---

## 八、实验任务

**实验目的：** 掌握定义和调用函数，创建和使用模块。  
**提交方式：** 在 Blackboard 系统提交一个文本文件 (txt 后缀)，文件中记录每道题的源程序和运行结果。

### 1. 二分查找

编写一个程序使用二分法查找给定的包含若干整数的列表 `s` 中是否存在给定的整数 `k`。使用二分查找的前提是列表已按照从小到大的顺序排序。为此，程序需要先判断 `s` 是否已经排好序。若未排好序，则需调用 `qsort` 函数进行排序并输出排序结果。

需要实现函数 `is_sorted` 和递归函数 `binary_search`。`binary_search` 在列表 `s` 的索引值属于闭区间 `[low, high]` 的元素中查找 `k`，若找到则返回 `k` 的索引值，否则返回 `-1`。

**程序 4.21（需补充完整）：**
```python
def is_sorted(s):  
    # to be implemented
    pass

def qsort(s):  
    if len(s) <= 1: return s
    s_less = []; s_greater = []; s_equal = []
    for k in s:
        if k < s[0]:
            s_less.append(k)
        elif k > s[0]:
            s_greater.append(k)
        else:
            s_equal.append(k)
    return qsort(s_less) + s_equal + qsort(s_greater)

def binary_search(s, low, high, k):  
    # to be implemented
    pass

s = [5, 6, 21, 32, 51, 60, 67, 73, 77, 99]
if not is_sorted(s):
    s = qsort(s)
    print(s)

print(binary_search(s, 0, len(s) - 1, 5))  
print(binary_search(s, 0, len(s) - 1, 31)) 
print(binary_search(s, 0, len(s) - 1, 99)) 
print(binary_search(s, 0, len(s) - 1, 64)) 
print(binary_search(s, 0, len(s) - 1, 51)) 
```

### 2. 有理数的四则运算

编写一个模块 `rational.py` 实现有理数的四则运算。有理数的一般形式是 $a/b$，其中 $a$ 是整数，$b$ 是正整数，并且当 $a$ 非 0 时 $|a|$ 和 $b$ 的最大公约数是 1。

程序中用一个列表 `[n, d]` 表示有理数，其中 `n` 表示分子，`d` 表示分母。
* `reduce` 函数调用 `gcd` 函数进行约分。
* 函数 `add`、`sub`、`mul` 和 `div` 分别进行加减乘除运算，运算的结果都需要约分，并且分母不出现负号。
* 函数 `output` 按照示例格式输出，例如 `[-13,12]` 输出为 `"-13/12"`。
* 函数 `get_rational` 从表示有理数的字符串中得到列表 `[n, d]`，例如从 `(-20/-3)` 得到 `[-20, 3]`。

用户在命令行输入三个命名参数：“`--op`”表示运算符（`add`、`sub`、`mul`、`div`），“`--x`”和“`--y`”表示进行计算的两个有理数（以字符串形式输入，圆括号括起）。

**程序 4.22（需补充完整）：**
```python
import sys, math

def test_all_functions():  
    # to be implemented
    pass

def gcd(a, b):  
    while a != b:
        if a > b:
            a -= b
        else:
            b -= a
    return a

def reduce(n, d):  
    # to be implemented
    pass

def add(x, y):  
    # to be implemented
    pass

def sub(x, y):  
    # to be implemented
    pass

def mul(x, y):  
    # to be implemented
    pass

def div(x, y):  
    # to be implemented
    pass

def output(x):  
    # to be implemented
    pass

def get_rational(s):  
    # to be implemented
    pass

if __name__ == '__main__':
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == '-h'):
        print(__doc__)
    elif len(sys.argv) == 2 and sys.argv[1] == 'test':
        test_all_functions()
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--op', type=str)
        parser.add_argument('--x', type=str)
        parser.add_argument('--y', type=str)
        args = parser.parse_args()
        
        op = args.op
        x = get_rational(args.x); y = get_rational(args.y)
        f = {'add':add, 'sub':sub, 'mul':mul, 'div':div}
        output(f[op](x, y))
```