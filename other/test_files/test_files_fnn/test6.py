import time

l1 = []
l2 = []
l3 = []
l4 = []
one = time.time()
for i in range(100000):
    l1.append(i)
two = time.time()
for i in range(100000):
    l2.insert(0, i)
three = time.time()
for i in range(100000):
    j = i * 3
    l3.append(j)
four = time.time()
for i in range(100000):
    l4.append(i * 3)
five = time.time()

print(two - one)
print(three - two)
print(four - three)
print(five - four)
