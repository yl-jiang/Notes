# 序列的修改、散列和切片

## 使自定义的对象支持切片操作

```python
class MySeq:

    def __getitem__(self, index):
        return index
```

测试上述程序：

```python
s = MySeq()

print(s[1])  # output: 1 / 正常

print(s[1:4])  #output: slice(1, 4, None) / 神奇的是Python将传入的切片形式自动转换为slice类型的对象

print(s[1:4:2])  # output: slice(1, 4, 2)

print(s[1:4:2, 9])  # output: (slice(1, 4, 2), 9) / 返回了一个元组，其中的一个元素为slice对象，另一个为整数

print(s[1:4:2, 1:3])  # output: (slice(1, 4, 2), slice(1, 3, None))
```

由此说明了，`my_seq[a:b:c]`句法背后的工作原理：创建`slice(a, b, c)`对象，交给 `__getitem__`方法处理。

这启发我们只需要在`__getitem__()`方法中，对传入的`index`进行一下类型判断，这样即可实现使用切片的方式访问自定义`class`的目的。

```python
import numbers
import reprlib

class Vector:
    typecode = 'd'

    def __init__(self, componments):
        print('call __init__()')
        self._componments = array(self.typecode, componments)

    def __iter__(self):
        print('call __iter__()')
        return iter(self._componments)

    def __repr__(self):
        print('call __repr__()')
        # reprlib.repr方法，可以生成长度有限的表示形式
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
        return tuple(self) == tuple(other)

    def __abs__(self):
        print('call __abs__()')
        return math.sqrt(sum(x**2 for x in self))

    def __bool__(self):
        print('call __bool__()')
        return bool(abs(self))

    def __len__(self):
        return len(self._components)

    def __getitem__(self, index):
        cls = type(self) # 获取实例所属的类
        if isinstance(index, slice):  # 如果取的是一个序列的数据，那么仍然返回一个class
            return cls(self._componments[index])
        elif isinstance(index, numbers.Integral):
            return self._componments[index]
        else:
            msg = '{cls.__name__} indices must be integers'
            raise TypeError(msg.format(cls=cls))

    @classmethod
    def frombytes(cls, octets):
        print('call classmethod frombytes()')
        typecode = chr(octets[0])
        memv = memoryview(octets[1:]).cast(typecode)
        return cls(memv)
```

测试上面的程序：

```python
v7 = Vector(range(7))  # output: call __init__()

print(v7[-1])  # output: 6.0

print(v7[1:4])  
# output: 
call __init__() 
call __repr__() 
Vector([1.0, 2.0, 3.0])

print(v7[1, 2])
# output: 
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-65-54e115c7e17b> in <module>
----> 1 v7[1, 2]

<ipython-input-60-551b698abfab> in __getitem__(self, index)
     47         else:
     48             msg = '{cls.__name__} indices must be integers'
---> 49             raise TypeError(msg.format(cls=cls))
     50 
     51     @classmethod

TypeError: Vector indices must be integers
```

---

## \_\_repr__()

调用`repr()`函数的目的在于调试，因此绝对不能抛出异常。如果`__repr__`方法实现有问题，那么必须要进行处理，且尽量输出有用的内容。

---

## 协议

在面向对象编程中，协议是非正式的接口，只在文档中定义，在代码中不定义。

例如，Python的序列协议只需要 `__len__` 和 `__getitem__` 两个方法。任何类（如 `Spam`），只要使用标准的签名和语义实现了这两个方法，就能用在任何期待序列的地方。`Spam` 是不是哪个类的子类无关紧要，只要提供了所需的方法即可。我们说它是序列，因为它的行为像序列，这才是重点。

---

## \_\_getattr\_\_()

使用`@property`装饰器可以方便的设置可访问的属性，但如果想访问的属性很多，这种方法就显得没有效率。Python中，属性查找失败后，解释器会调用`__getattr__`方法。简单来说，对`my_obj.x`表达式，Python会检查`my_obj`实例有没有名为`x`的属性(使用`@property`修饰的属性)； 如果没有，到类（`my_obj.__class__`）(`class`中定义的属性)中查找；如果还没有，顺着继承树继续查找。如果依旧找不到，调用`my_obj`所属类中定义的 `__getattr__`方法，传入`self`和属性名称的字符串形式（如 `'x'`）。

```python
class ToyClass:
    maybe_attrs = 'xyz'

    def __init__(self):
        self.ls = [1, 2, 3]

    def __getattr__(self, attr):
        cls = type(self)
        if len(attr) == 1:
            pos = cls.maybe_attrs.find(attr)
            if 0 <= pos < len(self.maybe_attrs):
                return self.ls[pos]
        msg = '{.__name__!r} object has no attribute {!r}'
        raise AttributeError(msg.format(cls, attr))
```

测试：

```python
t = ToyClass()
print(t.x)  # output: 1
t.x = 100  # 修改属性x的值
print(t.x)  # output: 100
print(t.ls)  # output: [1, 2, 3] / ls中的元素值并没有被改动
```

**使用`__getattr__`可以访问没有明确定义的属性，但是却无法修改**。这是由 `__getattr__`的运作方式导致的：仅当对象没有指定名称的属性时，Python才会调用那个方法，这是一种后备机制。可是，像 `t.x = 100`这样赋值之后，`t`对象就有`x`属性了，因此使用`t.x`获取`x`属性的值时不会再调用`__getattr__`方法了，解释器直接返回绑定到`t.x`上的值，即`100`。另一方面，`__getattr__`方法的实现没有考虑到`‘self.ls’`之外的实例属性，而是从这个属性中获取 `maybe_attrs` 中所列的“虚拟属性”。 

为了避免`__getattr__`方法的上述弊端，一般而言，实现了`__getattr__`方法的同时也需要实现`__setattr__`方法。

```python
def __setattr__(self, name, value):
    """
    禁止修改未事先定义的属性
    """
    cls = type(self)
    if len(name) == 1:
        if name in self.maybe_attrs:
            error = "readonly attribute {attr_name!r}"
        elif name.islower():
            error = "can't set attributes 'a' to 'z' in {cls_name!r}"
        else:
            error = ''
        if error:
            msg = error.format(cls_name=cls.__name__, attr_name=name)
            raise AttributeError(msg)
    super().__setattr__(name, value)
```

通过`__setattr__`方法，我们禁止给属性赋值的操作发生（也可以将实现逻辑改为`inplace`的修改`‘self.ls’`中对应位置的值）。在方法的最后我们使用`super()`函数，将子类的方法托管到超类中的相应方法上。
