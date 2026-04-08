# 人工智能数学原理与算法

**实验内容：分支和迭代**  
**日期：** 2025 年 3 月 16 日

---

## 目录
1. [分支](#一-分支)
   - if 语句和 if-else 表达式
   - 多分支 if 语句
2. [迭代](#二-迭代)
   - for 语句
   - while 语句
3. [推导式](#三-推导式)
4. [穷举](#四-穷举)
   - 求解 [100, 200] 的所有质数
   - 求解 3-sum 问题
   - 求解 subset-sum 问题
5. [实验任务](#五-实验任务)

---

## 一、分支

分支的语义是根据若干条件是否满足从多个分支中选择一个运行，由 `if` 语句和 `if-else` 表达式实现。

### 1. if 语句和 if-else 表达式

`if` 语句可以有多种形式，以计算整数的绝对值为例说明。根据绝对值的定义，可以有以下三种实现方式：

**程序 3.1：单分支 if 语句**
```python
x = int(input("Please enter an integer: "))
y = x
if x < 0:
    y = -x
print("The absolute value of %d is %d" %(x, y))
```

**程序 3.2：两分支 if 语句**
```python
x = int(input("Please enter an integer: "))
if x < 0:
    y = -x
else:
    y = x
print("The absolute value of %d is %d" %(x, y))
```

**程序 3.3：if-else 表达式**
```python
x = int(input("Please enter an integer: "))
y = -x if x < 0 else x
print("The absolute value of %d is %d" %(x, y))
```

### 2. 多分支 if 语句

`if` 语句包含的每个分支可以是一条语句，也可以是多条语句组成的语句块。这些分支相对 `if` 所在行必须有四个空格的缩进。

**程序 3.4：利用多分支 if 语句将百分制成绩转换为等级分**
```python
x = int(input("Please enter a score within [0, 100]: "))
grade = 'F'          
if x > 100 or x < 0:
    grade = 'Z'      
elif x >= 90:
    grade = 'A'      
elif x >= 80:
    grade = 'B'      
elif x >= 70:
    grade = 'C'      
elif x >= 60:
    grade = 'D'      
print("The grade of score %d is %c" %(x, grade))
```

---

## 二、迭代

迭代的语义是当某一条件满足时反复运行一个语句块，由 `for` 语句、`while` 语句和推导式实现。

### 1. for 语句

`for` 语句包含循环变量、可迭代对象和一个语句块。语句块相对 `for` 所在行必须有四个空格的缩进。

之前介绍的所有序列类型（包括 `list`、`tuple`、`range` 和 `str` 等）和 `set`、`dict` 等类型的对象称为**可迭代对象 (iterable)**，即可以通过 `for` 语句访问其包含的所有元素，在每次迭代时循环变量的值等于一个元素。

**程序 3.5：输出 1 到 10 之间的所有自然数的和**
```python
sum = 0; n = 10
for i in range(1, n+1):
    sum += i
print("The sum of 1 to %d is %d" % (n, sum))
```

**程序 3.6：输出 100 到 120 之间的所有偶数**
```python
lb = 100; ub = 120
for i in range(lb, ub+1, 2):
    print(i, end=' ')
```

**程序 3.7：输出一个由整数组成的集合中所包含的 3 的倍数**
```python
nums = {25, 18, 91, 365, 12, 78, 59}
for i in nums:
    if i % 3 == 0: print(i, end=' ')
```

**程序 3.8：输出集合中 3 的倍数 (`continue` 语句)**  
*注：`continue` 语句跳过本次循环的剩余语句并开始下一次循环。*
```python
nums = {25, 18, 91, 365, 12, 78, 59}
for i in nums:
    if i % 3 != 0: continue
    print(i, end=' ')
```

**程序 3.9：采用两种方式输出通讯录中的每个联系人姓名和电话号码**
```python
contacts = {"Tom":12345, "Jerry":54321, "Mary":23415}

for name, num in contacts.items():
    print('%s -> %d' % (name, num), end='; ')
print()
for name in contacts.keys():
    print('%s -> %d' % (name, contacts[name]), end='; ')
```

**程序 3.10：输出一个字符串中的所有字符和其对应的 Unicode 编码值**
```python
s = 'Python'
for c in s:
    print('(%s : %d)' % (c, ord(c)), end=' ')
```

**程序 3.11：输出一个列表中的所有元素的最大值和最小值**
```python
s = [21, 73, 6, 67, 99, 60, 77, 5, 51, 32]
max = s[0]; min = s[0]
for i in range(1, len(s)):
    if max < s[i]: max = s[i] 
    if min > s[i]: min = s[i]
print(max, min)  
```

### 2. while 语句

`for` 语句常用于循环次数已知的情形，而 `while` 语句也适用于循环次数未知的情形。

`while` 语句包含一个条件表达式和一个语句块。语句块相对 `while` 所在行必须有四个空格的缩进。`while` 语句的运行过程如下：
1. 对条件表达式求值。
2. 若值为 `False`，则 `while` 语句运行结束。
3. 若值为 `True`，则运行语句块，然后跳转到 1。

**程序 3.12：输出 1 到 10 之间的所有自然数的和**
```python
sum = 0; n = 10; i = 1
while i <= n:
    sum += i
    i += 1
print("The sum of 1 to %d is %d" % (n, sum))
```

**程序 3.13：输出 100 到 120 之间的所有偶数**
```python
i = lb = 100; ub = 120
while i <= ub:
    print(i, end=' ')
    i += 2
```

**程序 3.14：输出两个正整数的最大公约数**
```python
a = 156; b = 732
str = 'The greatest common divisor of %d and %d is ' % (a, b)
while a != b:
    if a > b:
        a -= b;
    else:
        b -= a;
print(str + ('%d' % a))
```

---

## 三、推导式

`list`、`dict` 和 `set` 等容器类型都提供了一种称为**推导式 (comprehension)** 的紧凑语法，可以通过迭代从已有容器创建新的容器。

**程序 3.15：推导式示例**
```python
nums = {25, 18, 91, 365, 12, 78, 59}
multiplier_of_3 = [n for n in nums if n % 3 == 0]
print(multiplier_of_3)  

square_of_odds = {n*n for n in nums if n % 2 == 1}
print(square_of_odds)   

s = [25, 18, 91, 365, 12, 78, 59, 18, 91]
sr = {n:n%3 for n in set(s)}
print(sr)  

tr = {n:r for (n,r) in sr.items() if r==0}
print(tr)  
```

---

## 四、穷举

穷举 (exhaustive search) 是一种解决问题的基本方法。穷举法的基本思想是：当问题的解属于一个规模较小的有限集合时，可以通过逐一列举和检查集合中的所有元素找到解。

### 1. 求解 [100, 200] 的所有质数

100 到 200 之间的所有质数都是自然数。解决本问题的方法是：列举 100 到 200 之间的所有自然数，逐一检查每个自然数是否是质数。

当问题比较复杂时，在编写程序之前应提出一个设计方案：
1. 列举给定区间内的每个自然数 $i$。
   1.1 对于 $i$，判断其是否质数。对于每个从 $2$ 到 $i - 1$ 的自然数 $j$：
       1.1.1 检查 $i$ 是否可以被 $j$ 整除。
   1.2 若存在这样的 $j$，则 $i$ 非质数。否则 $i$ 为质数，输出 $i$。

**算法优化：**
1. 在步骤 1 列举自然数时，只需列出奇数。
2. 在步骤 1.1 查找 $i$ 的因子 $j$ 时，$j$ 的取值范围的上界可以缩小为 $\sqrt{i}$。

**程序 3.16：求解质数（嵌套 for 语句）**
```python
import math
lb = 100; ub = 200
if lb % 2 == 0: lb += 1
if ub % 2 == 0: ub -= 1
for i in range(lb, ub + 1, 2):
    isPrime = True
    for j in range(2, math.ceil(math.sqrt(i)) + 1):
        if i % j == 0:
            isPrime = False
            break
    if isPrime: print(i, end=' ')
```

### 2. 求解 3-sum 问题

**问题描述：** 给定一个整数 $x$ 和一个由整数构成的集合 $S$，从 $S$ 中找一个由三个元素构成的子集，该子集中的三个元素之和必须等于 $x$。使用穷举法列举 $S$ 的所有由三个元素构成的子集，逐个检查其是否满足条件。

三重循环的循环变量 $i, j$ 和 $k$ 依次表示组成的子集的三个元素的索引值，它们满足严格递增关系。

**程序 3.17：3-sum 问题求解**
```python
S = [21, 73, 6, 67, 99, 60, 77, 5, 51, 32]
n = len(S)
x = 152
for i in range(n - 2):
    for j in range(i + 1, n - 1):
        for k in range(j + 1, n):
            if S[i] + S[j] + S[k] == x:
                print(S[i], S[j], S[k])  
```

### 3. 求解 subset-sum 问题

**问题描述：** 给定一个整数 $x$ 和一个由整数构成的集合 $S$，从 $S$ 中找一个子集，该子集中的所有元素之和必须等于 $x$。

设 $S$ 包含 $n$ 个元素，则 $S$ 的每个子集 $T$ 和 $n$ 位二进制数存在一一映射。$n$ 位二进制数的第 $k$ 位为 $1$ 表示第 $k$ 个元素在子集 $T$ 中，为 $0$ 则表示不在。
*例如：设 $S = \{1, 2, 3, 4\}$，$10$ 的二进制形式是 $1010$，其对应的子集是 $\{1, 3\}$。*

**程序 3.18：subset-sum 问题求解**
```python
S = [21, 73, 6, 67, 99, 60, 77, 5, 51, 32]
n = len(S)
x = 135
for i in range(1, 2 ** n):
    T = []
    for j in range(n-1, -1, -1):
        if (i >> j) % 2 == 1: T.append(S[n-1-j])
    if sum(T) == x: 
        print(T, end = ' ') 
```

---

## 五、实验任务

**实验目的：** 掌握分支和迭代的语句。  
**提交方式：** 在 Blackboard 系统提交一个文本文件 (txt 后缀)，文件中记录每道题的源程序和运行结果。

### 1. 考拉兹猜想
定义一个从给定正整数 $n$ 构建一个整数序列的过程如下。开始时序列只包含 $n$。如果序列的最后一个数 $m$ 不为 $1$，则根据 $m$ 的奇偶性向序列追加一个数：
* 如果 $m$ 是偶数，则追加 $m/2$；
* 否则追加 $3 \times m + 1$。

考拉兹猜想 (Collatz conjecture) 认为从任意正整数构建的序列都会以 $1$ 终止。

**要求：** 编写程序读取用户输入的正整数 $n$，然后在 `while` 循环中输出一个以 $1$ 终止的整数序列。输出的序列显示在一行，相邻的数之间用空格分隔。  
*例如：用户输入 `17` 得到的输出序列是 `17 52 26 13 40 20 10 5 16 8 4 2 1`。*

### 2. 字符串加密
编写程序实现基于偏移量的字符串加密。加密的过程是对原字符串中的每个字符对应的 Unicode 值加上一个偏移量，然后将得到的 Unicode 值映射到该字符对应的加密字符。

**要求：** 用户输入一个不小于 `-15` 的非零整数和一个由大小写字母或数字组成的字符串，程序生成并输出加密得到的字符串。  
*例如：用户输入 `10` 和字符串 `Attack at 1600`，得到的加密字符串是 `K~~kmu*k~*;@::`。*

### 3. 推导式转换为 for 语句
将 **程序 3.15** 中的所有推导式转换为 `for` 语句的形式。