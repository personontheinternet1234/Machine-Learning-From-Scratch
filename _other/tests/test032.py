def collect(nested):
    if isinstance(nested, list):
        return [collect(sublist) for sublist in nested]
    elif not isinstance(nested, list):
        return nested
    else:
        return None


list2 = [[1.1, 1.2], [1.3, 1.4]]
list3 = [[2.1, 2.2], [2.3, 2.4, [5, 6, 7]]]
list1 = [list2, list3]

all_list = collect(list1)
print(all_list)
