def is_sorted(s):
#待完成
    for i in range(1, len(s)):
        if s[i] < s[i-1]:
            return False
    return True

def qsort(s):
    if len(s) <= 1: return s
    s_less = []; s_greater = []; s_equal = []
    pivot = s[0]
    for k in s:
        if k < pivot: s_less.append(k)
        elif k > pivot: s_greater.append(k)
        else: s_equal.append(k)
    return qsort(s_less) + s_equal + qsort(s_greater)

def binary_search(s, low, high, k):
#待完成
    if low > high:
        return -1
    
    mid = (low + high) // 2
    if s[mid] == k:
        return mid
    elif s[mid] > k:
        return binary_search(s, low, mid - 1, k)
    else:
        return binary_search(s, mid + 1, high, k)

# 测试代码
s = [5, 6, 21, 32, 51, 60, 67, 73, 77, 99]
if not is_sorted(s):
    s = qsort(s)

print("排序后:", s)

targets = [5, 31, 99, 64, 51]
for t in targets:
    result = binary_search(s, 0, len(s) - 1, t)
    print(f"查找 {t}: 索引为 {result}")