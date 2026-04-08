"""
Module for performing arithmetic operations for rational numbers.

To run the module, user needs to supply three named parameters:
1. op stands for the operation:
    add for addition
    sub for subtraction
    mul for multiplication
    div for division
2. x stands for the first operand
3. y stands for the second operand

x and y must be enclosed in paired parentheses.

For example:

>>> run rational.py --op add --x (2/3) --y (-70/40)
-13/12
>>> run rational.py --op sub --x (-20/3) --y (120/470)
-976/141
>>> run rational.py --op mul --x (-6/19) --y (-114/18)
2/1
>>> run rational.py --op div --x (-6/19) --y (-114/-28)
-28/361
"""

import math, sys

def gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def reduce(n, d):
    """ 约分并处理负号（分母始终为正） """
    common = gcd(n, d)
    n //= common
    d //= common
    if d < 0:
        n, d = -n, -d
    return [n, d]

def add(x, y):
    # a/b + c/d = (ad + bc) / bd
    return reduce(x[0]*y[1] + y[0]*x[1], x[1]*y[1])

def sub(x, y):
    return reduce(x[0]*y[1] - y[0]*x[1], x[1]*y[1])

def mul(x, y):
    return reduce(x[0]*y[0], x[1]*y[1])

def div(x, y):
    return reduce(x[0]*y[1], x[1]*y[0])

def output(x):
    print(f"{x[0]}/{x[1]}")

def get_rational(s):
    """ 解析类似 '(2/3)' 的字符串 """
    s = s.strip('()')
    n, d = map(int, s.split('/'))
    return [n, d]

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print(__doc__)
    elif len(sys.argv) == 2 and sys.argv[1] == '-h':
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
