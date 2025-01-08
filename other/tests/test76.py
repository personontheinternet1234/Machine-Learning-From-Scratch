def do_replace(lst: list, itm: int) -> list:
    try:
        idx = lst.index(None)
        lst[idx] = itm
    except ValueError:
        lst.append(itm)
    return lst


item = 10
list1 = [1, 3, 6, None, 2, None, 5]

do_replace(list1, item)
print(list1)
