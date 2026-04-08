n = int(input("请输入偏移量 n (>-15): "))
s = input("请输入待加密的字符串: ")

encrypted_s = ""
for char in s:
    # 获取 Unicode，加上偏移量，转回字符
    new_char = chr(ord(char) + n)
    encrypted_s += new_char

print(f"加密后的字符串为: {encrypted_s}")
