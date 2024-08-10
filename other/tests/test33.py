def collect_numbers_preserve_shape(nested_list):
    if isinstance(nested_list, list):
        return [sublist for sublist in nested_list]
    elif isinstance(nested_list, (float, int)):
        return nested_list
    else:
        return None


list2 = [[1.1, 1.2], [1.3, 1.4]]
list3 = [[2.1, 2.2], [2.3, 2.4, [5, 6, 7]]]
list1 = [list2, list3]

all_numbers_with_shape = collect_numbers_preserve_shape(list1)
print(all_numbers_with_shape)