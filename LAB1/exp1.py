# 根据公式设置变量并进行计算

A = 1000000
years = 30
n = years * 12
rates = [0.04, 0.05, 0.06]

for r in rates:
    monthly_r = r / 12
    # 等额本息月供公式
    p = (monthly_r * A) / (1 - (1 + monthly_r)**(-n))
    total_payment = p * n
    print(f"利率 {r*100:.0f}%: 月供 {p:.2f}, 总还款 {total_payment:.2f}")
