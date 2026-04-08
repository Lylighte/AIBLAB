nums = {25, 18, 91, 365, 12, 78, 59}

# 1. 列表推导式转 for
multiplier_of_3 = []
for n in nums:
    if n % 3 == 0:
        multiplier_of_3.append(n)

print(multiplier_of_3)

# 2. 集合推导式转 for
square_of_odds = set()
for n in nums:
    if n % 2 == 1:
        square_of_odds.add(n * n)

print(square_of_odds)

s = [25, 18, 91, 365, 12, 78, 59, 18, 91]
# 3. 字典推导式转 for (sr)
sr = {}
for n in set(s):
    sr[n] = n % 3

print(sr)

# 4. 字典推导式转 for (tr)
tr = {}
for (n, r) in sr.items():
    if r == 0:
        tr[n] = r

print(tr)