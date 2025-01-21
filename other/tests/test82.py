import types


class MyClass:
    def __init__(self, value):
        self._value = value
        self._func = None

    def define_call(self):
        def func(x):
            return self._value * x
        self._func = func

    def __call__(self, x):
        return self._func(x)


obj = MyClass(5)

# obj.define_call()

print(obj(3))
