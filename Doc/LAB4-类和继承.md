# 人工智能数学原理与算法

**实验内容：类和继承**  
**日期：** 2025 年 3 月 16 日

---

## 目录
1. [定义和使用类](#一-定义和使用类)
   - 二维平面上的点
   - 一元多项式
2. [继承与覆盖](#二-继承与覆盖)
   - 近似计算一阶导数的公式
   - 继承
   - 覆盖
3. [实验任务](#三-实验任务)
   - 任务 1：表示有理数的类
   - 任务 2：定积分的数值计算

---

## 一、定义和使用类

面向对象的软件设计和开发技术是当前软件开发的主流技术，使得软件更易维护和复用。规模较大的软件大多是基于面向对象技术开发，以类作为基本组成单位。

在使用面向对象技术开发软件时，首先通过对软件需求的分析找到问题域中同一类的客观事物，称为**对象**。把对象共同的属性和运算封装在一起得到的程序单元就是**类**。类可作为一个独立单位进行开发和测试。以下举例说明。

### 1. 二维平面上的点

将二维平面上的点视为对象，抽象出其共同的属性和运算，定义一个类表示点。
* **点的属性**包括 $x$ 坐标、$y$ 坐标和名称（默认值为空串）。
* **点的运算**包括：
  * 给定坐标创建一个点
  * 沿 $x$ 轴平移
  * 沿 $y$ 轴平移
  * 以另一个点为中心旋转
  * 计算与另一个点之间的距离等

#### `Point2D` 类说明
* 类的定义由 `class 类名(父类)` 开始。
* 方法 `__init__` 称为**构造方法**，用来初始化新创建的对象的所有属性。
* 从一个类创建一个对象的语法是在类名后面加上一对圆括号，括号内可为空（表示无实参）或包含一些实参。如果实参的数量超过一个，它们之间用逗号分隔。这些实参必须和构造方法中除了 `self` 以外的那些形参在数量上相同，并且在顺序上一一对应。
* 定义在类内部的函数称为**方法**。
* 在一个类中的所有方法中出现的属性都需要使用 `对象名.属性` 进行限定。
* `self` 是一个特殊的对象名，表示**当前对象**。一个类中的所有方法的第一个形参都是 `self`。
* 对于一个对象调用其所属类的方法的语法是，在 `对象名.方法名` 后面加上一对圆括号，括号内可为空（表示无实参）或包含一些实参。如果实参的数量超过一个，它们之间用逗号分隔。这些实参必须和该方法中除了 `self` 以外的那些形参一一对应。
* Python 规定了类的一些特殊的方法，这些方法的名称都以双下划线 `__` 开始和结束。调用这些方法时可以使用简化语法。

**程序 5.1：`Point2D` 类的实现**

```python
import math

class Point2D:
    def __init__(self, x, y, name=''):
        self.x = x  
        self.y = y  
        self.name = name  

    def move_x(self, delta_x):
        self.x += delta_x

    def move_y(self, delta_y): 
        self.y += delta_y

    def rotate(self, p, t): 
        xr = self.x - p.x; yr = self.y - p.y
        x1 = p.x + xr * math.cos(t) - yr * math.sin(t)
        y1 = p.y + xr * math.sin(t) + yr * math.cos(t)
        self.x = x1; self.y = y1;

    def distance(self, p): 
        xr = self.x - p.x; yr = self.y - p.y
        return math.sqrt(xr * xr + yr * yr)

    def __str__(self): 
        if len(self.name) < 1:
            return '(%g, %g)' % (self.x, self.y)
        else:
            return '%s: (%g, %g)' % (self.name, self.x, self.y)

# 测试代码
a = Point2D(-5, 2, 'a')
print(a)  
a.move_x(-1); print(a)             
a.move_y(2); print(a)              
b = Point2D(3, 4, 'b')
print(b)  
print('The distance between a and b is %f' % a.distance(b))
b.rotate(a, math.pi/2)
print(a); print(b)     
a.rotate(b, math.pi)
print(a); print(b)     
```

### 2. 一元多项式

程序 5.4 定义了一个类 `Polynomial` 表示一元多项式。
* 多项式中的每一项的指数和其对应的系数存储在一个字典 `poly` 中。如果在运算结果中某一项的系数的绝对值小于一个预先定义的阈值 `tol`，则认为系数等于零，该项消失。
* 程序实现了多项式的加、乘、求值等运算。
* 程序定义了方法 `__call__`。对多项式 $p(x)$ 在 $x=t$ 时求值的语法简化 `p(t)` 被转换为与其等价的方法调用 `p.__call__(t)`。
* 程序定义了方法 `__str__`。该方法返回多项式的字符串表示，处理了多种特殊情形使得输出结果符合数学表达习惯。

**程序 5.4：`Polynomial` 类的实现**

```python
tol = 1E-15

class Polynomial:
    def __init__(self, poly):
        self.poly = {}
        for power in poly:
            if abs(poly[power]) > tol:
                self.poly[power] = poly[power]

    # call 功能：对多项式 p(x) 在 x=t 时求值的语法简化 p(t) 被转换为与其等价的方法调用 p.__call__(t)
    def __call__(self, x):
        value = 0.0
        for power in self.poly:
            value += self.poly[power]*x**power
        return value

    def __add__(self, other): 
        sum = self.poly.copy()   
        for power in other.poly:
            if power in sum:
                sum[power] += other.poly[power]
            else:
                sum[power] = other.poly[power]
        return Polynomial(sum) 

    def __mul__(self, other):
        sum = {}
        for self_power in self.poly:
            for other_power in other.poly:
                power = self_power + other_power
                m = self.poly[self_power] * other.poly[other_power]
                if power in sum:
                    sum[power] += m
                else:
                    sum[power] = m
        return Polynomial(sum)  

    def __str__(self):
        s = ''
        for power in sorted(self.poly):
            s += ' + %g*x^%d' % (self.poly[power], power)
        s = s.replace('+ -', '- ')
        s = s.replace('x^0', '1')
        s = s.replace(' 1*', ' ')
        s = s.replace('x^1 ', 'x ')
        if s[0:3] == ' + ':  
            s = s[3:]
        if s[0:3] == ' - ':  
            s = '-' + s[3:]
        return s

# 测试代码
p1 = Polynomial({0: -1, 2: 1, 7: 3}); print(p1)
p2 = Polynomial({0: 1, 2: -1, 5: -2, 3: 4}); print(p2)
p3 = p1 + p2; print(p3)
p4 = p1 * p2; print(p4)
print(p4(5))   
```

---

## 二、继承与覆盖

### 1. 近似计算一阶导数的公式

利用有限差分可以近似计算函数 $f(x)$ 的一阶导数。以下按照精确度从低到高次序列举公式，依次称为一阶向前差分、一阶向后差分、二阶中心差分和四阶中心差分：

$$ f'(x) = \frac{f(x + h) - f(x)}{h} + O(h) $$

$$ f'(x) = \frac{f(x) - f(x - h)}{h} + O(h) $$

$$ f'(x) = \frac{f(x + h) - f(x - h)}{2h} + O(h^2) $$

$$ f'(x) = \frac{4}{3} \frac{f(x + h) - f(x - h)}{2h} - \frac{1}{3} \frac{f(x + 2h) - f(x - 2h)}{4h} + O(h^4) $$

### 2. 继承

如果每个公式都用一个类实现，则这些类都有属性 `f` 和 `h`，并且它们的构造方法是相同的。对于每个公式，都需要比较其计算结果和精确结果的差别。这就导致了大量重复代码。

面向对象编程范式提供了**继承**机制，新类可从已有类获得属性和方法并进行扩展，实现了代码的重复利用。新类称为**子类**或派生类。已有类称为**父类**或基类。

可为这些公式类定义一个共同的父类 `Differentiation`，该类定义了初始化属性 `f` 和 `h` 的构造方法和比较结果的方法。通过继承，这些公式类可以获得这些属性和方法。 

**程序 5.5：导数计算类的继承实现**

```python
class Differentiation:
    def __init__(self, f, h=1E-5, dfdx_exact=None):
        self.f = f
        self.h = float(h)
        self.exact = dfdx_exact

    def get_error(self, x):  
        if self.exact is not None:
            df_numerical = self(x) 
            df_exact = self.exact(x) 
            return abs( (df_exact - df_numerical) / df_exact )

class Forward1(Differentiation):  
    def __call__(self, x):
        f, h = self.f, self.h
        return (f(x+h) - f(x))/h

class Backward1(Differentiation):  
    def __call__(self, x):
        f, h = self.f, self.h
        return (f(x) - f(x-h))/h

class Central2(Differentiation):  
    def __call__(self, x):
        f, h = self.f, self.h
        return (f(x+h) - f(x-h))/(2*h)

class Central4(Differentiation):  
    def __call__(self, x):
        f, h = self.f, self.h
        return (4./3)*(f(x+h) - f(x-h))  /(2*h) - \
               (1./3)*(f(x+2*h) - f(x-2*h))/(4*h)

def table(f, x, h_values, methods, dfdx=None):
    print('%-10s' % 'h', end=' ')
    for h in h_values: print('%-8.2e' % h, end=' ')
    print()
    for method in methods:
        print('%-10s' % method.__name__, end=' ')
        for h in h_values:
            if dfdx is not None:
                d = method(f, h, dfdx)
                output = d.get_error(x)
            else:
                d = method(f, h)
                output = d(x)
            print('%-8.6f' % output, end=' ')
        print()

import math
def g(x): return math.exp(x*math.sin(x))

import sympy as sym
sym_x = sym.Symbol('x')  
sym_gx = sym.exp(sym_x*sym.sin(sym_x)) 
sym_dgdx = sym.diff(sym_gx, sym_x) 
dgdx = sym.lambdify([sym_x], sym_dgdx)

table(f=g, x=-0.65, h_values=[10**(-k) for k in range(1, 7)],
      methods=[Forward1, Central2, Central4], dfdx=dgdx)
```

**程序 5.6：上述代码的输出结果**
```text
h          1.00e-01 1.00e-02 1.00e-03 1.00e-04 1.00e-05 1.00e-06
Forward1   0.104974 0.010906 0.001095 0.000110 0.000011 0.000001
Central2   0.004611 0.000046 0.000000 0.000000 0.000000 0.000000
Central4   0.000080 0.000000 0.000000 0.000000 0.000000 0.000000
```

### 3. 覆盖

子类可以对父类进行功能上的扩展，例如在子类中定义父类中没有的属性和方法。子类还可以重新定义从父类继承的方法，称为**覆盖 (overriding)**。覆盖的规则是子类中定义的某个方法和父类中的某个方法在名称、形参列表和返回类型上都相同，但方法体不同。

若一个子类定义了构造方法，则子类定义的构造方法覆盖了其父类的构造方法，此时若子类如果仍然需要从其父类继承属性，子类的构造方法必须包含语句 `super().__init__()`；若一个子类未定义构造方法，则继承了其父类的构造方法，因此自动从其父类继承属性。

**程序 5.7：覆盖特性的演示**

```python
class Parent:
    def __init__(self):
        self.c = 1
    def m(self):
        print('Calling m in class Parent')
	
class Child1(Parent):
    def __init__(self):  
        super().__init__()  
        self.d = 2
    def m(self):  
        print('Calling m in class Child1')
	
class Child2(Parent):
    def m(self):  
        super().m()  
        print('Calling m in class Child2')

class Child3(Parent):
    def __init__(self):  
        self.f = 3
    def m2(self):
        print('Calling m2 in class Child3')	
	
c1 = Child1()           
c2 = Child2()           
c3 = Child3()           
p = Parent()            
c1.m()                  
c2.m()                  
c3.m()                  
p.m()                   
print(c1.__dict__)      
print(c2.__dict__)      
print(c3.__dict__)      
```

---

## 三、实验任务

**实验目的：** 掌握类的定义和继承，掌握重载和覆盖的使用。  
**提交方式：** 在 Blackboard 系统提交一个文本文件 (txt 后缀)，文件中记录每道题的源程序和运行结果。

### 1. 表示有理数的类

有理数的一般形式是 $a/b$，其中 $a$ 是整数，$b$ 是正整数，并且当 $a$ 非 0 时 $|a|$ 和 $b$ 的最大公约数是 1。实现 `Rational` 类表示有理数和其运算。

* 程序 5.8 已列出了部分代码，需要实现空缺的函数。
* `Rational` 类的属性 `nu` 和 `de` 分别表示分子和分母。
* 函数 `__add__`、`__sub__`、`__mul__` 和 `__truediv__` 分别进行加减乘除运算，然后返回一个新创建的 `Rational` 对象作为运算结果。
* 函数 `__eq__`、`__ne__`、`__gt__`、`__lt__`、`__ge__` 和 `__le__` 比较两个有理数，返回一个 bool 类型的值。这些函数对应的比较运算符分别是：`==`、`!=`、`>`、`<`、`>=`、`<=`。例如表达式 `Rational(6, -19) > Rational(14, -41)` 在求值时被转换成方法调用 `Rational(6, -19).__gt__(Rational(14, -41))`。
* 函数 `test` 测试这些函数。`gcd` 函数要求形参 $a$ 和 $b$ 都是正整数，如果其中出现 0 或负数，递归不会终止。

**程序 5.8（需补充完整）：**

```python
def gcd(a, b):  
    while a != b:
        if a > b:
            a -= b
        else:
            b -= a
    return a

class Rational:
    def __init__(self, n=0, d=1):  
        _nu = n; _de = d
        self.__dict__['nu'] = _nu; self.__dict__['de'] = _de

    def __setattr__(self, name, value):
        raise TypeError('Error: Rational objects are immutable')

    def __str__(self): return '%d/%d' % (self.nu, self.de)

    def __add__(self, other):  
        # to be implemented
        pass

    def __sub__(self, other):  
        # to be implemented
        pass

    def __mul__(self, other):  
        # to be implemented
        pass

    def __truediv__(self, other):  
        # to be implemented
        pass

    def __eq__(self, other):  
        # to be implemented
        pass

    def __ne__(self, other):  
        # to be implemented
        pass

    def __gt__(self, other):  
        # to be implemented
        pass

    def __lt__(self, other):  
        # to be implemented
        pass

    def __ge__(self, other):  
        # to be implemented
        pass

    def __le__(self, other):  
        # to be implemented
        pass

def test():
    testsuite = [
        ('Rational(2, 3) + Rational(-70, 40)',
          Rational(-13, 12)),
        ('Rational(-20, 3) - Rational(120, 470)',
          Rational(-976,141)),
        ('Rational(-6, 19) * Rational(-114, 18)',
          Rational(2, 1)),
        ('Rational(-6, 19) / Rational(-114, -28)',
          Rational(-28,361)),

        ('Rational(-6, 19) == Rational(-14, 41)', False),
        ('Rational(-6, 19) != Rational(-14, 41)', True),
        ('Rational(6, -19) > Rational(14, -41)', True),
        ('Rational(-6, 19) < Rational(-14, 41)', False),
        ('Rational(-6, 19) >= Rational(-14, 41)', True),
        ('Rational(6, -19) <= Rational(14, -41)', False),
        ('Rational(-15, 8) == Rational(120, -64)', True),
    ]
    for t in testsuite:
        try:
            result = eval(t[0])
        except:
            print('Error in evaluating ' + t[0]); continue

        if result != t[1]:
            print('Error:  %s != %s' % (t[0], t[1]))

if __name__ == '__main__':
    test()
```

### 2. 定积分的数值计算

函数 $f(x)$ 在区间 $[a, b]$ 上的定积分可用区间内选取的 $n + 1$ 个点 $x_i$ ($i = 0, 1, ..., n$)（称为积分节点）上的函数值的加权和近似计算：

$$ \int_{a}^{b} f(x) dx \approx \sum_{i=0}^{n} w_i f(x_i) $$

其中 $w_i$ 是函数值 $f(x_i)$ 的权值，称为积分系数。不同的数值计算公式的区别体现在积分节点和积分系数上。

**表：定积分的几种数值计算公式**

| 公式名称 | 积分节点的坐标和积分系数 |
| :--- | :--- |
| **复合梯形公式** | $x_i = a + ih \quad \text{for} \quad i = 0, ..., n, \quad h = \frac{b-a}{n}$ <br> $w_0 = w_n = \frac{h}{2}, \quad w_i = h \quad \text{for} \quad i = 1, ..., n - 1$ |
| **复合辛普森公式** <br> (*$n$ 必须是偶数*<br>*若输入的 $n$ 是奇数，则执行 $n = n + 1$*) | $x_i = a + ih \quad \text{for} \quad i = 0, ..., n, \quad h = \frac{b-a}{n}$ <br> $w_0 = w_n = \frac{h}{3}, \quad w_i = \frac{2h}{3} \quad \text{for} \quad i = 2, 4, ..., n - 2$ <br> $w_i = \frac{4h}{3} \quad \text{for} \quad i = 1, 3, ..., n - 1$ |
| **复合高斯-勒让德公式** <br> (*$n$ 必须是奇数*<br>*若输入的 $n$ 是偶数，则执行 $n = n + 1$*) | $x_i = a + \frac{i+1}{2}h - \frac{\sqrt{3}}{6}h \quad \text{for} \quad i = 0, 2, ..., n - 1$ <br> $x_i = a + \frac{i}{2}h + \frac{\sqrt{3}}{6}h \quad \text{for} \quad i = 1, 3, ..., n$ <br> $h = \frac{2(b-a)}{n+1}, \quad w_i = \frac{h}{2} \quad \text{for} \quad i = 0, 1, ..., n$ |

在**程序 5.9** 中实现 `Integrator` 类的 `integrate` 方法和它的三个子类，分别对应表中的三种公式。在每个子类中只需覆盖父类的 `compute_points` 方法计算并返回两个列表，它们分别存储了所有积分节点的坐标和积分系数。

`test()` 函数用函数 $f(x) = (x \cos x + \sin x)e^{x \sin x}$ 和它的解析形式的积分函数 $F(x) = e^{x \sin x}$ 测试这三个公式的精确度。

**程序 5.9（需补充完整）：**

```python
import math

class Integrator:
    def __init__(self, a, b, n):
        self.a, self.b, self.n = a, b, n
        self.points, self.weights = self.compute_points()

    def compute_points(self):
        raise NotImplementedError(self.__class__.__name__)

    def integrate(self, f):  
        # to be implemented
        pass

class Trapezoidal(Integrator):
    def compute_points(self):  
        # to be implemented
        pass

class Simpson(Integrator):
    def compute_points(self):  
        # to be implemented
        pass

class GaussLegendre(Integrator):
    def compute_points(self):  
        # to be implemented
        pass

def test():
    def f(x): return (x * math.cos(x) + math.sin(x)) * \
                      math.exp(x * math.sin(x))
    def F(x): return math.exp(x * math.sin(x))

    a = 2; b = 3; n = 200
    I_exact = F(b) - F(a)
    tol = 1E-3

    methods = [Trapezoidal, Simpson, GaussLegendre]
    for method in methods:
        integrator = method(a, b, n)
        I = integrator.integrate(f)
        rel_err = abs((I_exact - I) / I_exact)
        print('%s: %g' % (method.__name__, rel_err))
        if rel_err > tol:
            print('Error in %s' % method.__name__)

if __name__ == '__main__':
    test()
```