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
import io
from contextlib import redirect_stdout

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

def test_all_functions():
    total = 0
    passed = 0

    def check(name, actual, expected):
        nonlocal total, passed
        total += 1
        if actual == expected:
            passed += 1
            print(f"[PASS] {name}: {actual}")
        else:
            print(f"[FAIL] {name}: expected={expected}, actual={actual}")

    check('gcd(54, 24)', gcd(54, 24), 6)
    check('gcd(-18, 30)', gcd(-18, 30), 6)

    check('reduce(10, 20)', reduce(10, 20), [1, 2])
    check('reduce(-8, -12)', reduce(-8, -12), [2, 3])
    check('reduce(4, -6)', reduce(4, -6), [-2, 3])

    check("get_rational('(2/3)')", get_rational('(2/3)'), [2, 3])
    check("get_rational('(-70/40)')", get_rational('(-70/40)'), [-70, 40])

    check('add([2, 3], [-70, 40])', add([2, 3], [-70, 40]), [-13, 12])
    check('sub([-20, 3], [120, 470])', sub([-20, 3], [120, 470]), [-976, 141])
    check('mul([-6, 19], [-114, 18])', mul([-6, 19], [-114, 18]), [2, 1])
    check('div([-6, 19], [-114, -28])', div([-6, 19], [-114, -28]), [-28, 361])

    buf = io.StringIO()
    with redirect_stdout(buf):
        output([1, 2])
    check('output([1, 2])', buf.getvalue().strip(), '1/2')

    print(f"Summary: {passed}/{total} passed")

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
