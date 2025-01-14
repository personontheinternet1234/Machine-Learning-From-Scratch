items = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
for i, itm in enumerate(items):
    print(i)
    itm.append('modified')
print(items)
