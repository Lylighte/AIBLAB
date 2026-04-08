def gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

class Rational:
    def __init__(self, n=0, d=1):
        common = gcd(n, d)
        self.__dict__['nu'] = n // common if d > 0 else -n // common
        self.__dict__['de'] = abs(d // common)

    def __setattr__(self, name, value):
        raise TypeError('Error: Rational objects are immutable')

    def __str__(self): 
        return '%d/%d' % (self.nu, self.de)

    # 算术运算
    def __add__(self, other):
        return Rational(self.nu * other.de + other.nu * self.de, self.de * other.de)

    def __sub__(self, other):
        return Rational(self.nu * other.de - other.nu * self.de, self.de * other.de)

    def __mul__(self, other):
        return Rational(self.nu * other.nu, self.de * other.de)

    def __truediv__(self, other):
        return Rational(self.nu * other.de, self.de * other.nu)

    # 比较运算
    def __eq__(self, other):
        return self.nu == other.nu and self.de == other.de

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        return self.nu * other.de > other.nu * self.de

    def __lt__(self, other):
        return self.nu * other.de < other.nu * self.de

    def __ge__(self, other):
        return not self < other

    def __le__(self, other):
        return not self > other

def test():
    testsuite = [
        ('Rational(2, 3) + Rational(-70, 40)', Rational(-13, 12)),
        ('Rational(-20, 3) - Rational(120, 470)', Rational(-976,141)),
        ('Rational(-6, 19) * Rational(-114, 18)', Rational(2, 1)),
        ('Rational(-6, 19) / Rational(-114, -28)', Rational(-28,361)),

        ('Rational(-6, 19) == Rational(-14, 41)', False),
        ('Rational(-6, 19) != Rational(-14, 41)', True),
        ('Rational(6, -19) > Rational(14, -41)', True),
        ('Rational(-6, 19) < Rational(-14, 41)', False),
        ('Rational(-6, 19) >= Rational(-14, 41)', True),
        ('Rational(6, -19) <= Rational(14, -41)', False),
        ('Rational(-15, 8) == Rational(120, -64)', True),
        
        ('Rational(1, 2) + Rational(1, 3) * Rational(3, 4)', Rational(1, 2)),
        ('(Rational(1, 2) + Rational(1, 3)) * Rational(3, 4)', Rational(5, 8)),
    ]
    for t in testsuite:
        try:
            result = eval(t[0])
        except:
            print('Error in evaluating ' + t[0]); continue

        if result != t[1]:
            print('Error:  %s != %s' % (t[0], t[1]))
        else:
            print('OK:  %s == %s' % (t[0], t[1]))

if __name__ == '__main__':
    test()