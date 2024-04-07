import numpy as np


def l_relu(values):
    output = np.maximum(0.1 * values, values)
    return output


def d_l_relu(values):
    return np.where(values > 0, 1, 0.1)


def t_break():
    print("---------------")


# [0, 2]
t_c = 0

# x
bob = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]

# y
diana = [np.array([350, 514]), np.array([712, 1068]), np.array([1024, 1550])]

# W
jeff = [np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]), np.array([[4, 6], [5, 8], [2, 3], [5, 7]])]

# B
arnold = [np.array([5, 6, 7, 8]), np.array([4, 9])]

# f
f_a = [bob]
f_s = [bob[t_c]]
for i in range(len(jeff)):
    f_a.append(l_relu(np.matmul(f_a[i], jeff[i]) + arnold[i]))
    f_s.append(l_relu(np.matmul(f_s[i], jeff[i]) + arnold[i]))

# b
d_a_a = [np.multiply(-2, np.subtract(diana, f_a[-1]))]
d_w_a = []
d_b_a = []

d_a_s = [np.multiply(-2, np.subtract(diana[t_c], f_s[-1]))]
d_w_s = []
d_b_s = []

for i in range(-1, -len(f_a) + 1, -1):
    print(i)
    # d_b_a.insert(0, d_l_relu(np.matmul(jeff[i], f_a[i - 1]) + arnold[i]) * d_a_a[0])
for i in range(-1, -len(f_s) + 1, -1):
    d_b_s.insert(0, d_l_relu(np.matmul(jeff[i], f_s[i - 1]) + arnold[i]) * d_a_s[0])

print(d_b_a)
t_break()
print(d_b_s)

# r
# for i in f_a:
#     print(i)
# t_break()
# for i in f_s:
#     print(i)
