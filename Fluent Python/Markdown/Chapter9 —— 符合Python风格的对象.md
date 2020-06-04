

# 符合Python风格的对象

如何使用特殊方法和约定的结构，定义行为良好且符合Python风格的类。 

```python
class Vector2d:
    typecode = 'd'

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    # 把Vector2d的实例变成可迭代对象
    def __iter__(self):
        print('call __iter__()')
        return (i for i in (self.x, self.y))

    def __repr__(self):
        print('call __repr__()')
        class_name = type(self).__name__
        # 因为Vector2d的实例是可迭代对象，所以*self会把x和y提供给format函数
        return '{}({!r}, {!r})'.format(class_name, *self)

    def __str__(self):
        print('call __str__()')
        return str(tuple(self)) 

    def __bytes__(self):
        print('call __bytes__()')
        return (bytes([ord(self.typecode)]) + 
                bytes(array(self.typecode, self)))

    def __eq__(self, other):
        print('call __eq__()')
        # 这种比较方式存在弊端，如：Vector2d(3, 4) == [3, 4] 为True
        return tuple(self) == tuple(other)

    def __abs__(self):
        print('call __abs__()')
        return math.hypot(self.x, self.y)

    def __bool__(self):
        print('call __bool__()')
        return bool(abs(self))

    # 被classmethod装饰器装饰的方法，只能由class本身调用而不是实例
    @classmethod  
    def frombytes(cls, octets):
        typecode = chr(octets[0])
        memv = memoryview(octets[1:]).cast(typecode)
        print('memv: ', *memv)
        return cls(*memv)

    # 如果没有实现该方法，那么当使用format(class_instance)时，默认使用__str__()的结果
    def __format__(self, fmt_spec=''):
        components = (format(c, fmt_spec) for c in self)
        return '({}, {})'.format(*components)
```

测试上述代码：

```python
In[]: bytes([ord('d')])  # output: b'd'

In[]: vv1 = Vector2d(3, 4)
In[]: print(v1.x, v1.y)  # output: 3.0 4.0

In[]: x, y = v1  # output: call __iter__()
In[]: print(x, y)  # output: 3.0 4.0

In[]: print(v1)  # output: call __str__() ; call __iter__(); (3.0, 4.0)

In[]: v1_clone = eval(repr(v1))  # output: call __repr__(); call __iter__()

In[]: print(v1 == v1_clone)  # output: call __eq__(); call __iter__(); call __iter__(); True

In[]: binary = bytes(v1)  # output: call __bytes__(); call __iter__()
In[]: print(binary)  #output: b'd\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@'

In[]: print(bool(v1))  #output: call __bool__(); call __abs__(); True

In[]: print(Vector2d(3, 4) == [3, 4])  # output: call __eq__(); call __iter__(); True

In[]: vx = Vector2d.frombytes(binary)  #output: memv:  3.0 4.0

print(format(v2))  #output: call __iter__(); '(3.0, 4.0)'
print(format(v2, '.2f'))  #output: call __iter__(); '(3.00, 4.00)'

```

---

## 私有属性

在Python中，把`class`代码里面以双下划线开头的变量作为私有属性，私有属性会保存在`class`的`__dict__`属性中，私有属性在`__dict__`中的命名规则是`'_classname__PrivateAttributeName'`。

**私有属性的目的是防止意外的直接访问，但不能防止有意的破坏。**

```python
class A:

    def __init__(self, x):
        # __x为私有属性
        self.__x = x
```

测试：

```python
test = A(5)
print(test.__dict__)
print(test.__x)
```

输出结果：

```Python
{'_A__x': 5}

---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-143-40521ab1a009> in <module>
----> 2 print(test.__x)

AttributeError: 'A' object has no attribute '__x'
```

如果知道访问的正确机制，任何人还是都可以直接读取私有属性：

```python
print(test._A__x)  #output: 5
```

**`mymod.py`中前缀为下划线的名称不会被导入，然而，依旧可以使用`from mymod import _privatefunc`将其导入。**

**Python程序员默认使用以单下划线开头的属性作为私有属性，但这只是互相的约定，并没有Python内部机制的加持。**

---

## 类属性：\_\_dict\_\_

**Python在各个实例中名为`__dict__`的字典存储实例属性。因为字典使用了底层的散列表以提升访问速度，因此字典会消耗大量内存。如果要处理拥有数百万个属性的实例，这种方法就显得不太合适，解决这一问题可以使用`__slots__`类属性。**

---

## 类属性：\_\_slots\_\_

**让解释器在元组中存储实例属性，而不用字典，从而可以节省大量内存。**

+ **`__slots__`属性不能通过继承获得**，Python只会使用个各类中定义的`__slots__`属性

定义`__slots__`的方式是，创建一个类属性，使用`__slots__`这个名字，并把它的值设为一个字符串构成的可迭代对象，其中各个元素表示各个实例属性。我喜欢使用元组，因为这样定义的`__slots__`中所含的信息不会变化。

```python
class Vector2d:
    __slots__ = ('__x', '__y')
    ...
    pass
```

**在类中定义`__slots__`属性的目的是告诉解释器：“这个类中的所有实例属性都在这儿了！”这样，Python 会在各个实例中使用类似元组的结构存储实例变量，从而避免使用消耗内存的`__dict__`属性。如果有数百万个实例同时活动，这样做能节省大量内存。**

### 值得注意的

+ 在类中定义`__slots__`属性之后，实例不能再有`__slots__`中所列名称之外的其他属性。这只是一个副作用，不是`__slots__`存在的真正原因。
+ 不要使用`__slots__`属性禁止类的用户新增实例属性。`__slots__`是用于优化的，不是为了约束程序员。
+ 如果想把实例作为弱引用的目标，则必须有`__weakref__`这个属性。一般情况下，用户定义的类中默认就包含`__weakref__`属性。如果你使用了`__slots__`，那么一定要将`__weakref__`添加到`__slots__`中
+ **如果想修改类属性的值，必须直接在类上修改，不能通过实例修改**
+ **类属性是公开的，因此会被子类继承**
