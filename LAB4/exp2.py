import math

class Integrator:
    def __init__(self, a, b, n):
        self.a, self.b, self.n = a, b, n
        self.points, self.weights = self.compute_points()

    def integrate(self, f):
        return sum(w * f(x) for x, w in zip(self.points, self.weights))

class Trapezoidal(Integrator):
    # 复合梯形公式：等距点，权重为h，首尾权重为h/2
    def compute_points(self):
        n, a, b = self.n, self.a, self.b
        h = (b - a) / n
        x = [a + i*h for i in range(n + 1)]
        w = [h] * (n + 1)
        w[0] = w[n] = h / 2
        return x, w

class Simpson(Integrator):
    # 复合辛普森公式：等距点，权重为h/3，奇数点权重为4h/3，偶数点权重为2h/3
    def compute_points(self):
        if self.n % 2 != 0: self.n += 1
        n, a, b = self.n, self.a, self.b
        h = (b - a) / n
        x = [a + i*h for i in range(n + 1)]
        w = [0] * (n + 1)
        w[0] = w[n] = h / 3
        for i in range(1, n):
            w[i] = (4*h/3) if i % 2 != 0 else (2*h/3)
        return x, w

class GaussLegendre(Integrator):
    # 复合高斯-勒让德公式：非等距点，权重根据表格计算
    # n 必须是奇数
    def compute_points(self):
        if self.n % 2 == 0: self.n += 1
        n, a, b = self.n, self.a, self.b
        h = 2 * (b - a) / (n + 1)
        x_pts, w_pts = [], []
        sqrt3_6 = math.sqrt(3) / 6
        # 按照表中的公式
        for i in range(0, n, 2):
            x_pts.append(a + (i+1)/2 * h - sqrt3_6 * h)
        for i in range(1, n + 1, 2):
            x_pts.append(a + i/2 * h + sqrt3_6 * h)
        
        x_pts.sort() # 排序以匹配顺序
        w_pts = [h/2] * len(x_pts)
        return x_pts, w_pts

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
