# 重载运算符

+ 不能重载内置类型的运算符
+ 不能新建运算符，只能重载现有的
+ 某些运算符不能重载——`is`, `and`, `or` 和`not`
+ 遵循运算符的一个基本规则：始终返回一个新对象。也就是说，不能修改`self`（不能进行inplace的修改），要创建并返回合适类型的新实例

---

```python
import numbers
import operator
import functools
import itertools
from array import array
import reprlib

class Vector:
    typecode = 'd'
    shortcut_names = 'xyzt'

    def __init__(self, componments):
        print('call __init__()')
        self._componments = array(self.typecode, componments)

    def __iter__(self):
        print('call __iter__()')
        return iter(self._componments)

    def __repr__(self):
        print('call __repr__()')
        componments = reprlib.repr(self._componments)
        componments = componments[componments.find('['):-1]
        return 'Vector({})'.format(componments)

    def __str__(self):
        print('call __str__()')
        return str(tuple(self))

    def __bytes__(self):
        print('call __bytes__()')
        return (bytes([ord(self.typecode)]) + bytes(self._componments))

    def __eq__(self, other):
        print('call __eq__()')
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))

    def __abs__(self):
        print('call __abs__()')
        return math.sqrt(sum(x**2 for x in self))

    def __bool__(self):
        print('call __bool__()')
        return bool(abs(self))

    def __len__(self):
        print('call __len__()')
        return len(self._componments)

    def __getitem__(self, index):
        print('call __getitem__()')
        cls = type(self) # 获取实例所属的类
        if isinstance(index, slice):  # 如果取的是一个序列的数据，那么仍然返回一个class
            return cls(self._componments[index])
        elif isinstance(index, numbers.Integral):
            return self._componments[index]
        else:
            msg = '{cls.__name__} indices must be integers'
            raise TypeError(msg.format(cls=cls))

    def __getattr__(self, name):
        print('call __getattr__()')
        cls = type(self)
        if len(name) == 1:
            pos = cls.shortcut_names.find(name)
            if 0 <= pos < len(self._componments):
                return self._componments[pos]
        msg = '{.__name__!r} object has no attribute {!r}'
        raise AttributeError(msg.format(cls, name))

    def __setattr__(self, name, value):
        print('call __setattr__()')
        cls = type(self)
        if len(name) == 1:
            if name in cls.shortcut_names:
                error = 'readonly attribute {attr_name!r}'
            elif name.islower():
                error = "can't set attributes 'a' to 'z' in {cls_name!r}"
            else:
                error = ''

            if error:
                msg = error.format(cls_name=cls.__name__, attr_name=name)
                raise AttributeError(msg)

        super().__setattr__(name, value)

    def __hash__(self):  # 跟之前一样，自定义对象的hash值由其所有属性的异或运算得到
        print('call __hash__()')
        hashes = (hash(x) for x in self._componments)
        return functools.reduce(operator.xor, hashes, 0)

    def angle(self, n):
        r = math.sqrt(sum(x**2 for x in self[n:]))
        a = math.atan2(r, self[n-1])
        if (n == len(self) - 1) and (self[-1] < 0):
            return math.pi * 2 - a
        else:
            return a

    def angles(self):
        return (self.angle(n) for n in range(1, len(self)))

    def __format__(self, fmt_spec=''):
        if fmt_spec.endswith('h'):
            fmt_spec = fmt_spec[:-1]
            coords = itertools.chain([abs(self)], self.angles())
            outer_fmt = '<{}>'
        else:
            coords = self
            outer_fmt = '({})'
        components = (format(c, fmt_spec) for c in coords)
        return outer_fmt.format(', '.join(components))

    @classmethod
    def frombytes(cls, octets):
        print('call classmethod frombytes()')
        typecode = chr(octets[0])
        memv = memoryview(octets[1:]).cast(typecode)
        return cls(memv)
```

## 一元运算符

### 重载一元运算符`'+'`

```python
v = Vector([1, 2, 3])
# []out:
# call __init__()
# call __setattr__()

def __pos__(self):
    print('call __pos__()')
    return Vector(self)

if __name__ == '__main__':
    Vector.__pos__ = __pos__
    print(+v)
```

输出：

```python
call __pos__()
call __init__()
call __iter__()
call __setattr__()
call __str__()
call __iter__()
call __len__()
(1.0, 2.0, 3.0)
```

### 重载一元运算符`'-'`

```python
def __neg__(self):
    print('call __neg__()')
    return Vector(-x for x in self)

if __name__ == '__main__':
    Vector.__neg__ = __neg__
    print(-v)
```

输出：

```python
call __neg__()
call __iter__()
call __init__()
call __setattr__()
call __str__()
call __iter__()
call __len__()
(-1.0, -2.0, -3.0)
```

---

## 中缀运算符（二元运算符）

### `__add__()`

```python
def __add__(self, other):
    """
    这样实现的__add__()方法，要求左操作数必须为Vector对象，右操作数可以为任意可迭代的对象。
    """
    print('call __add__()')
    pairs = itertools.zip_longest(self, other, fillvalue=0.)
    return Vector(a+b for a, b in pairs)


if __name__ == '__main__':
    Vector.__add__ = __add__
    print(v + [11, 22, 33, 44, 55])
```

输出：

```python
call __add__()
call __iter__()
call __init__()
call __setattr__()
call __str__()
call __iter__()
call __len__()
(12.0, 24.0, 36.0, 44.0, 55.0)
```

测试：

```Python
if __name__ == '__main__':
    print([11, 22, 33, 44, 55] + v)
```

输出：

```Python
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-86-dcbcf84211cf> in <module>
----> 1 print([11, 22, 33, 44, 55] + v)

TypeError: can only concatenate list (not "Vector") to list
```

我的理解是：使用`__add__()`方法时，Python会检查左操作数的类型，只有在确定左操作数是本`class`的一个实例时，才会执行。

### `__radd__()`

`__radd__` 方法是一种后备机制，如果左操作数没有实现 `__add__` 方法，或者实现了，但是返回 `NotImplemented` ，向解释器表明它不知道如何处理右操作数，那么 Python 会调用 `__radd__` 方法。

执行:

```Python
a + b
```

时，Python解释器会按照如下行为执行相应操作：

+ 如果 a 有 `__add__` 方法，而且返回值不是 `NotImplemented`，调用 `a.__add__(b)`，然后返回结果。 
+ 如果 a 没有 `__add__` 方法，或者调用 `__add__` 方法返回 `NotImplemented`，检查 b 有没有 `__radd__` 方法，如果有，而且没有返回 `NotImplemented`，调用 `b.__radd__(a)`，然后返回结果。
+ 如果 b 没有 `__radd__` 方法，或者调用 `__radd__` 方法返回 `NotImplemented`，则抛出 `TypeError`，并在错误消息中指明操作数类型不支持。

```Python
def __radd__(val1, val2):
    # 该方法是__add__()的后备方法。当左操作数不是Vector类型时，会尝试调用该方法
    print('call __radd__()')
    return val1 + val2

if __name__ == '__main__':
    Vector.__radd__ = __radd__
    print([11, 22, 33, 44, 55] + v)
```

输出：

```Python
call __radd__()
call __add__()
call __iter__()
call __init__()
call __setattr__()
call __str__()
call __iter__()
call __len__()
(12.0, 24.0, 36.0, 44.0, 55.0)
```

`class Vector`的`__add__()`方法仍有缺陷，例如，不能处理`Vector`和不可迭代类型数据的相加操作，也不能处理，与可迭代但非数值类型的对象进行相加操作。当执行这些未定义的操作时，返回的错误提示往往词不达意。对于这些未定义操作，应该设法返回有意义（`NotImplemented`）的错误提示。

例如：

```Python
if __name__ == '__main__':
    print(v + 1)
```

输出：

```Python
call __add__()
call __iter__()
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-89-f66d2396cea7> in <module>
----> 1 print(v + 1)

<ipython-input-85-e77c8c2914c7> in __add__(self, other)
      2     # 这样实现的__add__()方法，要求左操作数必须为Vector对象，右操作数可以为任意可迭代对象
      3     print('call __add__()')
----> 4     pairs = itertools.zip_longest(self, other, fillvalue=0.)
      5     return Vector(a+b for a, b in pairs)
      6 

TypeError: zip_longest argument #2 must support iteration
```

修正`__add__()`，当遇到未定义的操作时，抛出适当的异常信息:

```python 
def __add__(self, other):
    try:
        pairs = itertools.zip_longest(self, other, fillvalue=0.)
        return Vector(a+b for a, b in pairs)
    except TypeError:
        return NotImplemented

def __radd__(self, other):
    return self + other

if __name__ == '__main__':
    Vector.__add__ = __add__
    Vector.__radd__ = __radd__

    print(v + 1)
```

输出：

```python
call __iter__()
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-96-f66d2396cea7> in <module>
----> 1 print(v + 1)

TypeError: unsupported operand type(s) for +: 'Vector' and 'int'
```

### `__mul__()`

```python
def __mul__(self, scalar):
    return Vector(x*scalar for x in self)

def __rmul__(self, scalar):
    """
    该方法是__mul__()的后备方法。当左操作数不是Vector类型时，会尝试调用该方法
    """
    return self * scalar


if __name__ == '__main__':
    Vector.__mul__ = __mul__
    Vector.__rmul__ = __rmul__

    print(v * 10)
```

输出：

```Python
call __iter__()
call __init__()
call __setattr__()
call __str__()
call __iter__()
call __len__()
(10.0, 20.0, 30.0)
```

可为`__mul__()`增加类型检查，使得该方法的任务更明确：

```Python
def __mul__(self, scalar):
    if isinstance(scalar, numbers.Real):
        return Vector(x*scalar for x in self)
    else:
        return NotImplemented

def __rmul__(self, scalar):
    return self * scalar


if __name__ == '__main__':
    Vector.__mul__ = __mul__
    Vector.__rmul__ = __rmul__

    print(v * 'a')
```

输出：

```python
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-100-309c54ab453f> in <module>
     11 Vector.__rmul__ = __rmul__
     12 
---> 13 print(v * 'a')

TypeError: can't multiply sequence by non-int of type 'Vector'
```

### `__matmul__()`

```python
def __matmul__(self, other):
    try:
        return sum(a*b for a, b in zip(self, other))
    except TypeError:
        return NotImplemented

def __rmatmul__(self, other):
    return self @ other


if __name__ == '__main__':
    Vector.__matmul__ = __matmul__
    Vector.__rmatmul__ = __rmatmul__

    v1 = Vector([1,2,3])
    v2 = Vector([10, 10, 10])
    print(v1 @ v2)
```

输出：

```python 
call __init__()
call __setattr__()
call __init__()
call __setattr__()
call __iter__()
call __iter__()
60.0
```

如果中缀运算符的正向方法（如`__mul__`）**只处理与`self`属于同一类型的操作数**，那就无需实现反向方法（如`__rmul__()`），因为按照定义，**反向方法是为了处理类型不同的操作数**。

### `__eq__()`

```python
def __eq__(self, other):
    if isinstance(other, Vector):
        return (len(self) == len(other) and
               all(a == b for a, b in zip(self, other)))
    else:
        return NotImplemented

Vector.__eq__ = __eq__

class Vector2d:
    typecode = 'd'

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    # 把Vector2d的实例变成可迭代对象
    def __iter__(self):
        print('call __iter__() of Vector2d')
        return (i for i in (self.x, self.y))

    def __repr__(self):
        print('call __repr__() of Vector2d')
        class_name = type(self).__name__
        # 因为Vector2d的实例是可迭代对象，所以*self会把x和y提供给format函数
        return '{}({!r}, {!r})'.format(class_name, *self)

    def __str__(self):
        print('call __str__() of Vector2d')
        return str(tuple(self))

    def __bytes__(self):
        print('call __bytes__() of Vector2d')
        return (bytes([ord(self.typecode)]) + 
                bytes(array(self.typecode, self)))

    def __eq__(self, other):
        print('call __eq__() of Vector2d')
        return tuple(self) == tuple(other)

    def __abs__(self):
        print('call __abs__() of Vector2d')
        return math.hypot(self.x, self.y)

    def __bool__(self):
        print('call __bool__() of Vector2d')
        return bool(abs(self))

    # classmethod装饰器装饰的方法，只能由class本身调用而不是其实例
    @classmethod  
    def frombytes(cls, octets):
        typecode = chr(octets[0])
        memv = memoryview(octets[1:]).cast(typecode)
        print('memv: ', *memv)
        return cls(*memv)

    def angle(self):
        return math.atan2(self.y, self.x)

    def __format__(self, fmt_spec=''):
        # 如果传入的格式化字符串是以'p'结尾，那么就使用极坐标表示
        if fmt_spec.endswith('p'):
            fmt_spec = fmt_spec[:-1]
            coords = (abs(self), self.angle())
            out_fmt = '<{}, {}>'
        else:
            coords = self
            out_fmt = '({}, {})'
        components = (format(c, fmt_spec) for c in coords)
        return out_fmt.format(*components)
```

测试：

```python
def __eq__(self, other):
    if isinstance(other, Vector):
        return (len(self) == len(other) and
               all(a == b for a, b in zip(self, other)))
    else:
        return NotImplemented


if __name__ == '__main__':
    Vector.__eq__ = __eq__
    v_2d = Vector2d(1, 2)
    v_nd = Vector([1, 2])
    print(v_nd == v_2d)
```

输出：

```python
call __init__()
call __setattr__()
call __eq__() of Vector2d
call __iter__() of Vector2d
call __iter__()
call __len__()
True
```

**为什么`Vector`的`'__eq__()'`方法加了类型检查，上式还是判断为`True`？**

1. 为了计算`v_nd == v_2d`，Python首先调用`Vector.__eq__(v_nd, v_2d)`
2. 经`Vector.__eq__(v_nd, v_2d)`确认，`v_2d`不是`Vector`的实例，因此返回`NotImplemented`
3. Python得到`NotImplemented`结果，尝试调用`class Vector2d`的`__eq__()`方法，计算`tuple(v_2d) == tuple(v_nd)`，结果返回`True`

```python
v_nd_2 = Vector([1, 2, 3])
print(v_nd_2 == v_2d)
```

输出：

```python
call __init__()
call __setattr__()
call __eq__() of Vector2d
call __iter__() of Vector2d
call __iter__()
call __len__()
False
```

1. 为了计算`v_nd == v_2d`，Python首先调用`Vector.__eq__(v_nd, v_2d)`
2. 经`Vector.__eq__(v_nd, v_2d)`确认，`v_2d`不是`Vector`的实例，因此返回`NotImplemented`
3. Python得到`NotImplemented`结果，尝试调用`class Vector2d`的`__eq__()`方法，计算`tuple(v_2d) == tuple(v_nd_2)`，结果返回`False`

### `__ne__()`

当我们实现了`==`运算符后，就不再需要实现`!=`运算符了，因为Python会自动启用后备行为，使用`__eq__()`返回的结果的反面作为`__ne__()`的返回值。

### 就地增量运算符

如果一个`class`没有实现`inplace`运算符，增量赋值运算符只是语法糖:`a += b`的作用与`a = a + b`完全一样。而且，如果定义了`__add__()`方法的话，不用额外编写代码，就能使用`+=`运算符了。

对于不可变类型来说，实现`__add__()`等运算符就足够了，但对于可变类型来说实现`__iadd__()`等就地运算符可以更有效率。

**就地增量运算符必须返回`self`。**

```python
if __name__ == '__main__':
    v1 = Vector([1,2,3])
    v1_alias = v1
    print(id(v1), id(v1_alias))
    v1 += v1
    print(v1)
```

输出：

```python
call __init__()
call __setattr__()
---------------------
(140461673498896, 140461673498896)
---------------------
call __iter__()
call __iter__()
call __init__()
call __setattr__()
---------------------
call __getattr__()
call __getattr__()
call __getattr__()
call __getattr__()
call __repr__()
call __getattr__()
call __getattr__()
call __getattr__()
call __getattr__()
call __getattr__()
call __getattr__()
call __getattr__()
call __getattr__()
call __getattr__()
call __getattr__()
call __getattr__()
call __getattr__()
call __getattr__()
call __getattr__()
call __getattr__()
call __getattr__()
call __getattr__()
call __getattr__()
Vector([2.0, 4.0, 6.0])
```
