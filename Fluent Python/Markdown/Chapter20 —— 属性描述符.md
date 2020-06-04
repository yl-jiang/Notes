# 属性描述符

描述符是对多个属性运用相同存取逻辑的一种方式。描述符的典型用途——管理数据属性。

描述符是实现了特定协议的类，这个协议包括 `__get__`、`__set__` 和 `__delete__` 方法。`property`类实现了完整的描述符协议。通常，可以只实现部分协议。

除了特性之外，使用描述符的 Python 功能还有方法及 `classmethod` 和 `staticmethod` 装饰器。理解描述符是精通 Python 的关键。

实现了 `__get__`、`__set__` 或 `__delete__` 方法的类是描述符。

**描述符的用法是，创建一个实例，作为另一个类的类属性。**

```python
class Quantity:
    """
    由于我们将值存储在托管实例中，因此，无需实现__get__方法，就可以取得托管类对应特性的值。
    """

    def __init__(self, storage_name):
        self.storage_name = storage_name

    def __set__(self, instance, value):
        """
        尝试为托管属性赋值时，会调用__set__方法。

        params:
            self: 描述符实例，即LineItem.weight或LineItem.price;
            instance: 托管实例，即LineItem实例;
            value: 要设定的值。

        也就是说真正的value是存储在托管实例（LineItem）中的。
        """
        if value > 0:
            # 这里，因为属性名与托管类的特姓名相同，因此必须直接处理托管实例的__dict__属性，
            # 如果使用内置的setattr函数【setattr(instance, self.storage_name, value)】，
            # 会再次触发__set__方法，导致无限递归
            instance.__dict__[self.storage_name] = value
        else:
            raise ValueError('value must be > 0')
```

 也就是由于`weight = Quantity('weight')`我们将描述符实例名（也即，LineItem的特性）设为和托管实例属性名相同，如果使用`'instance.weight = xxx'`（或`setattr(instance, 'weight', value)`）时，又会触发描述符类的`__set__`方法。

 ```python
class LineItem:

    weight = Quantity('weight')
    price = Quantity('price')

    def __init__(self, description, weight, price):
        self.description = description
        self.weight = weight
        self.price = price

    def subtotal(self):
        return self.weight * self.price
 ```

 测试：

 ```python
>>> truffle = LineItem('white truffle', 100, 0)
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-3-96e1c0dce33a> in <module>
----> 1 truffle = LineItem('white truffle', 100, 0)

<ipython-input-2-04b79630509f> in __init__(self, description, weight, price)
      7         self.description = description
      8         self.weight = weight
----> 9         self.price = price
     10 
     11     def subtotal(self):

<ipython-input-1-24af9f20f958> in __set__(self, instance, value)
     22             instance.__dict__[self.storage_name] = value
     23         else:
---> 24             raise ValueError('value must be > 0')

ValueError: value must be > 0
 ```

 编写 `__set__` 方法时，要记住 `self` 和 `instance` 参数的意思：

+ `self` 是描述符实例；
+ `instance` 是托管实例

 管理实例属性的描述符应该把值存储在托管实例中。因此，Python 才为描述符中的那个方法提供了 `instance` 参数。

上述使用的描述符类有一个很大的缺陷，就是，每次实例化一个描述符类时，都需要输入托管属性的名称，当类似属性多起来，管理维护这些托管属性将变得复杂且容易出错。

我们想修改上述程序，以避免在描述符声明语句中重复输入属性名，实现该功能的主要思想是为每个托管类中的描述符属性生成一个全局唯一的名称。在此之前，我们要确保生成的变量名不同寻常。

```python
class T:

    def __init__(self):
        pass
```

```python
# 正常的Python句法无法做到创建一个名为”_q#3“的变量，因为`#`之后被Python当做注释
>>> _q#3 = 10
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-13-d0f7128bfef7> in <module>
      1 # 正常的Python句法无法做到创建一个名为”_q#3“的变量，因为`#`之后被Python当做注释
----> 2 _q#3 = 10

NameError: name '_q' is not defined
```

配合使用`setattr`和`getattr`可以创建和读取不平常的变量名。

```python
>>> setattr(T, '_q#3', 10)
>>> getattr(T, '_q#3')
10
```

接下来，开始实现我们最初的想法：

```python
class Quantity:

    """
    主要思想是：在描述符类中维护一个类属性计数器，每当我们使用描述符类时，更新该类属性的值，
    并借助该描述符类属性生成一个全局唯一的名称，并为对应托管实例创建一个以该名称命名的实例属性。
    """

    # __counter用于统计Quantity实例数量
    __counter = 0

    def __init__(self):
        cls = self.__class__
        prefix = cls.__name__
        index = cls.__counter
        self.storage_name = f'_{prefix}#{index}'
        cls.__counter += 1

    def __get__(self, instance, owner):
        # instance：托管类的实例
        # owner: 托管类
        return getattr(instance, self.storage_name)

    def __set__(self, instance, value):
        if value > 0:
            # 借助setattr()函数将值存储在instance中,因为存储的托管属性名称与托管实例的名称不一样
            setattr(instance, self.storage_name, value)
        else:
            raise ValueError('value must be > 0')
```

```python
class LineItem:

    weight = Quantity()
    price = Quantity()

    def __init__(self, description, weight, price):
        self.description = description
        self.weight = weight
        self.price = price

    def subtotal(self):
        return self.weight * self.price
```

测试：

```python
>>> coconuts = LineItem('Brazilian coconut', 20, 17.95)
>>> coconuts.weight, coconuts.price
(20, 17.95)
>>> getattr(coconuts, '_Quantity#0'), getattr(coconuts, '_Quantity#1')
(20, 17.95)
```

`__get__` 方法有三个参数：`self`、`instance` 和 `owner`。`owner` 参数是托管类（如 `LineItem`）的引用，通过描述符从托管类中获取属性时用得到。如果使用 `LineItem.weight` 从类中获取托管属性（以 `weight` 为例），描述符的 `__get__` 方法接收到的 `instance` 参数值是 `None`（因为此时LineItem还没有实例化，`__get__`函数中的`instance`就为`None`）。因此，下述控制台会话才会抛出 `AttributeError` 异常：

```python
>>> LineItem.weight
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-8-5b8ff6540ea2> in <module>
----> 1 LineItem.weight

<ipython-input-4-d10a62dfaaef> in __get__(self, instance, owner)
     19         # instance：托管类的实例
     20         # owner: 托管类
---> 21         return getattr(instance, self.storage_name)
     22 
     23     def __set__(self, instance, value):

AttributeError: 'NoneType' object has no attribute '_Quantity#0'
```

使用`'LineItem.weight'`抛出的异常说明语句可能会让使用者无法理解，应该修改抛出异常的说明信息，以更加直观。但是实现这种功能需要用到元编程，在这里，我们只做一点简单的改进：

```python
class Quantity:

    __counter = 0

    def __init__(self):
        cls = self.__class__
        prefix = cls.__name__
        index = cls.__counter
        self.storage_name = f'_{prefix}#{index}'
        cls.__counter += 1

    def __get__(self, instance, owner):
        if instance is None:
            # 如果不是通过实例调用，返回描述符自身
            return self
        else:
            return getattr(instance, self.storage_name)

    def __set__(self, instance, value):
        if value > 0:
            setattr(instance, self.storage_name, value)
        else:
            raise ValueError('value must be > 0')
```

```python
class LineItem:

    weight = Quantity()
    price = Quantity()

    def __init__(self, description, weight, price):
        self.description = description
        self.weight = weight
        self.price = price

    def subtotal(self):
        return self.weight * self.price
```

测试：

```python
>>> LineItem.price
<__main__.Quantity at 0x7f17b4530fd0>
```

---

## 函数属性

```python
def f(x):
    try:
        f.a += 1
    except:
        f.a = 0
    print('f.a', f.a)
    return None
```

```python
>>> a = f(4)
f.a 0
>>> b = f(7)
f.a 1
# 函数对象中始终保存着函数属性
>>> f.a
1
# a中存储的是函数f的返回值，而非函数对象
>>> a.a
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-18-a78c10d1b27f> in <module>
      1 # a中存储的是函数f的返回值，而非函数对象
----> 2 a.a

AttributeError: 'NoneType' object has no attribute 'a'
```

利用函数属性以及闭包，可以使用特性工厂函数达到无需为每次手动输入创建的特性名称的目的。

```python
def quantity():
    try:
        # 维持一个共享的函数属性，以确保使得能为每个特性生成一个全局唯一的名称
        quantity.counter += 1
    except AttributeError:
        quantity.counter = 0

    storage_name = f'quantity:{quantity.counter}'

    def qty_getter(instance):
        return getattr(instance, storage_name)

    def qty_setter(instance, value):
        if value > 0:
            setattr(instance, storage_name, value)
        else:
            raise ValueError('value must be > 0')

    # 每当使用quantity()，由于property的作用，会维持该quantity()的闭包
    return property(qty_getter, qty_setter)
```

```python
class LineItem:

    # 使用特性工厂函数，构建类特性
    weight = quantity()
    price = quantity()

    def __init__(self, description, weight, price):
        self.description = description
        # 此时weight已经是类特性，这里表示给类特性赋值
        self.weight = weight
        self.price = price

    def subtotal(self):
        # 获取类特性的值完成计算
        return self.weight * self.price
```

```python
>>> coconuts = LineItem('Brazilian coconut', 20, 17.95)
>>> coconuts.weight, coconuts.price
(20, 17.95)
>>> coconuts = LineItem('Brazilian coconut', -20, 17.95)
>>> coconuts.weight, coconuts.price
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-22-b5bb41cd59b8> in <module>
----> 1 coconuts = LineItem('Brazilian coconut', -20, 17.95)
      2 coconuts.weight, coconuts.price

<ipython-input-20-a30cc1a230bb> in __init__(self, description, weight, price)
      8         self.description = description
      9         # 此时weight已经是类特性，这里表示给类特性赋值
---> 10         self.weight = weight
     11         self.price = price
     12 

<ipython-input-19-a5ae41d70c4e> in qty_setter(instance, value)
     14             setattr(instance, storage_name, value)
     15         else:
---> 16             raise ValueError('value must be > 0')
     17 
     18     # 每当使用quantity()，由于property的作用，会维持该quantity()的闭包

ValueError: value must be > 0
```

---

## 更通用的抽象实现

描述符在类中定义，因此可以利用继承重用部分代码来创建新描述符。

描述符类的关键优势：通过子类共享代码，构建具有部分相同功能的专用描述符。

```python
import abc

class AutoStorage:

    __counter = 0

    def __init__(self):
        cls = self.__class__
        index= cls.__counter
        prefix = cls.__name__
        self.storage_name = f'_{prefix}#{index}'
        cls.__counter += 1

    def __get__(self, instance, owner):
        if instance is not None:
            return getattr(instance, self.storage_name)
        else:
            return self

    def __set__(self, instance, value):
        setattr(instance, self.storage_name, value)


class Validated(abc.ABC, AutoStorage):

    def __set__(self, instance, value):
        value = self.validated(value)
        super().__set__(instance, value)

    # 抽象方法，继承Validated的子类必须重写该方法
    @abc.abstractmethod
    def validated(self, value):
        """return validated value or raise ValueError."""
```

其他代码都是共享的，我们只需实现各托管属性的判断逻辑即可。

```python
class Quantity(Validated):

    def validated(self, value):
        if value > 0:
            return value
        else:
            raise ValueError('value must be > 0')


class NonBlank(Validated):

    def validated(self, value):
        value = value.strip()
        if len(value) == 0:
            raise ValueError('value cannot be empty or blank')
        return value
```

```python
class LineItem:

    description = NonBlank()
    weight = Quantity()
    price = Quantity()

    def __init__(self, description, weight, price):
        self.description = description
        self.weight = weight
        self.price = price

    def subtotal(self):
        return self.weight * self.price
```

测试：

```python
>>> coconuts = LineItem('Brazilian coconut', 20, 17.95)
>>> coconuts.weight, coconuts.price
(20, 17.95)
>>> coconuts = LineItem('', 20, 17.95)
>>> coconuts.weight, coconuts.price
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-26-819742ec439f> in <module>
----> 1 coconuts = LineItem('', 20, 17.95)
      2 coconuts.weight, coconuts.price

<ipython-input-24-db7b11dbe815> in __init__(self, description, weight, price)
      6 
      7     def __init__(self, description, weight, price):
----> 8         self.description = description
      9         self.weight = weight
     10         self.price = price

<ipython-input-23-5f46feb5c2d2> in __set__(self, instance, value)
     25 
     26     def __set__(self, instance, value):
---> 27         value = self.validated(value)
     28         super().__set__(instance, value)
     29 

<ipython-input-23-5f46feb5c2d2> in validated(self, value)
     48         value = value.strip()
     49         if len(value) == 0:
---> 50             raise ValueError('value cannot be empty or blank')
     51         return value

ValueError: value cannot be empty or blank
```

---

## 覆盖型与非覆盖型描述符对比

一些用于展示结果的函数：

```python
def cls_name(obj_or_cls):
    cls = type(obj_or_cls)
    if cls is type:
        cls = obj_or_cls
    return cls.__name__.split('.')[-1]

def display(obj):
    cls = type(obj)
    if cls is type:
        return f'<class {obj.__name__}>'
    elif cls in [type(None), int]:
        return repr(obj)
    else:
        return f'<{cls_name(obj)} object>'

def print_args(name, *args):
    pseudo_args = ', '.join(display(x) for x in args)
    print(f'-> {cls_name(args[0])}.__{name}__({pseudo_args})')
```

定义不同类型的描述符：

```python
class Overriding:
    """
    数据描述符，或强制描述符。
    """
    def __get__(self, instance, owner):
        print_args('get', self, instance, owner)

    def __set__(self, instance, value):
        print_args('set', self, instance, value)

class OverridingNoGet:
    """
    没有__get__方法的覆盖型描述符。
    """

    def __set__(self, instance, value):
        print_args('set', self, instance, value)

class NonOverriding:
    """
    非数据描述符或遮盖型描述符。
    """

    def __get__(self, instance, owner):
        print_args('get', self, instance, owner)
```

```python
class Managed:
    over = Overriding()
    over_no_get = OverridingNoGet()
    non_over = NonOverriding()

    def spam(self):
        print(f'-> Managed.spam({display(self)})')
```

实现了`__set__`方法的描述符为覆盖型描述符，因为实现`__set__`方法的话，会覆盖对实例属性的赋值操作，特性也是覆盖型描述符，没有实现`__set__`方法的描述符为非覆盖型描述符。

### 实现了`__get__`和`__set__`的覆盖型描述符

```python
>>> obj = Managed()
# obj.over触发描述符的__get__方法
>>> obj.over
-> Overriding.__get__(<Overriding object>, <Managed object>, <class Managed>)
```

```python
>>> Managed.over
-> Overriding.__get__(<Overriding object>, None, <class Managed>)
```

```python
# 为obj.over赋值，触发描述符的__set__方法
>>> obj.over = 7
-> Overriding.__set__(<Overriding object>, <Managed object>, 7)

# 使用实例的__dict__属性可跳过描述符为实例属性赋值
>>> obj.__dict__['over'] = 7
>>> vars(obj)
{'over': 7}

# 即便我们通过obj.__dict__['over'] = 7，为obj创建了一个名为over的属性， 
# 但是描述符（Managed.obj）仍然会覆盖读取obj.over这个操作
>>> obj.over
-> Overriding.__get__(<Overriding object>, <Managed object>, <class Managed>)
```

### 没有实现`__get__`方法的覆盖型描述符

```python
>>> obj = Managed()
# 由于没有实现__get__方法，因此obj.over_no_get从类中获取描述符实例
>>> obj.over_no_get
<__main__.OverridingNoGet at 0x7f17b458c3d0>
>>> Managed.over_no_get
<__main__.OverridingNoGet at 0x7f17b458c3d0>
```

```python
# 为描述符属性赋值，会触发描述符的__set__方法
>>> obj.over_no_get = 7
-> OverridingNoGet.__set__(<OverridingNoGet object>, <Managed object>, 7)

# 绕过描述符创建实例属性over_no_get
>>> obj.__dict__['over_no_get'] = 9
>>> vars(obj)
{'over_no_get': 9}

# 由于没有实现__get__，因此会直接从实例属性中取值
>>> obj.over_no_get
9

# 为obj.over_no_get赋值，仍然经过描述符的__set__方法处理
>>> obj.over_no_get = 7
-> OverridingNoGet.__set__(<OverridingNoGet object>, <Managed object>, 7)
# 但是读取时，只要有同名的实例属性，描述符就会被遮盖
>>> obj.over_no_get
9
```

### 非覆盖型描述符

```python
>>> obj = Managed()
# obj.non_over 触发描述符的 __get__ 方法，第二个参数的值是 obj
>>> obj.non_over
-> NonOverriding.__get__(<NonOverriding object>, <Managed object>, <class Managed>)
```

```python
>>> Managed.non_over
# 由于Managed.non_over是非覆盖型描述符，因此，描述符没有干涉赋值操作
>>> obj.non_over = 7
```

```python
# Managed.non_over 描述符依然存在，会通过类截获这次访问
>>> Managed.non_over
-> NonOverriding.__get__(<NonOverriding object>, None, <class Managed>)
```

```python
# 由于没有实现__set__方法，因此，描述符没有名为non_over的描述符属性，
# 此时，实例属性non_over会把Managed类的同名描述符属性遮盖掉
>>> obj.non_over
7
```

```python
# 如果把实例属性obj.non_over删除了，那么，读取obj.non_over时，会触发描述符的__get__方法
>>> del obj.non_over
>>> obj.non_over
-> NonOverriding.__get__(<NonOverriding object>, <Managed object>, <class Managed>)
```

---

## 绕过覆盖型描述符

读**类属性**的操作可以由依附在托管类上定义有 `__get__` 方法的描述符处理，但是写**类属性**的操作不会由依附在托管类上定义有 `__set__` 方法的描述符处理。

通过实例的`__dict__`无法达到绕过覆盖型描述符赋值的目的（虽然实际上不是，但呈现出来给用户的是这样），不管描述符是不是覆盖型，**为类属性赋值**都能覆盖描述符：

```python
>>> obj = Managed()

# 手动创建类属性，此时，即使是实现了__set__的描述符也无法插手对类属性的赋值
>>> Managed.over = 1
>>> Managed.over_no_get = 2
>>> Managed.non_over = 3

# 使用实例访问类属性
>>> obj.over, obj.over_no_get, obj.non_over
(1, 2, 3)
```

上述绕过覆盖型描述符的方法其实是利用实例属性查找顺序实现的。`'obj.attr'`这样的表达式不会从`obj`（实例）开始寻找`attr`，而是从`'obj.__class__'`开始，而且，仅当类中没有名为`attr`的特性时，Python才会在`obj`实例中寻找。

---

## 方法是描述符

```python
>>> obj = Managed()

# obj.spam获取的是'绑定方法'对象
>>> obj.spam
<bound method Managed.spam of <__main__.Managed object at 0x7f17b44d9f50>>

# Managed.spam获取的是函数
>>> Managed.spam
<function __main__.Managed.spam(self)>

>>> obj.spam()
-> Managed.spam(<Managed object>)
>>> Managed.spam(Managed)
-> Managed.spam(<class Managed>)
```

下面用一个简单的小例子说明上述现象的本质：

```python
import collections

class Text(collections.UserString):

    def __repr__(self):
        return 'Text({!r})'.format(self.data)

    def reverse(self):
        return self[::-1]
```

```python
>>> word = Text('forward')
>>> word
Text('forward')
```

实例（`word`）调用`reverse()`方法时，`reverse()`方法中的`self`参数代表了`Text`的某个实例，
Python会默认将实例`word`绑定到`reverse()`的`self`参数上。

```python
>>> word.reverse()
Text('drawrof')
```

class对象调用`reverse()`时，`reverse()`就相当于普通的函数，相应地，其中的`self`就被当做普通的参数。

```python
>>> Text.reverse('forward')
'drawrof'
```

```python
# 一个是函数，一个是方法
>>> type(Text.reverse), type(word.reverse)
(function, method)
```

用户定义的函数都有 `__get__` 方法，所以依附到类上(在类中定义)时，就相当于描述符，函数没有实现 `__set__` 方法，因此是非覆盖型描述符。

与描述符一样，通过托管类访问时，函数的 `__get__` 方法会返回自身的引用。但是，通过实例访问时，函数的 `__get__` 方法返回的是绑定方法对象：一种可调用的对象，里面包装着函数，并把托管实例（例如 `obj`）绑定给函数的第一个参数（即 `self`），这与 `functools.partial` 函数的行为一致。

函数都是非覆盖型描述符。**在函数上调用 `__get__` 方法时传入实例**，得到的是绑定到那个实例上的方法。

```python
>>> Text.reverse.__get__(word)
<bound method Text.reverse of Text('forward')>
```

调用函数的 `__get__` 方法时，如果`__get__`方法的`instance` 参数的值是 `None`，那么得到的是函数本身。

```python
>>> Text.reverse.__get__(None, Text)
<function __main__.Text.reverse(self)>
>>> Text.reverse.__get__(word, Text)
<bound method Text.reverse of Text('forward')>
```

`word.reverse` 表达式其实会调用 `Text.reverse.__get__(word)`，返回对应的绑定方法。

```python
>>> word.reverse
<bound method Text.reverse of Text('forward')>
```

绑定方法对象有个 `__self__` 属性，其值是调用这个方法的实例引用。

```python
>>> word.reverse.__self__
Text('forward')
>>> word.__repr__.__self__
Text('forward')
```

绑定方法的 `__func__` 属性是依附在托管类上那个原始函数的引用。

```python
>>> word.reverse.__func__ is Text.reverse
True
```

绑定方法对象还有个 `__call__` 方法，用于处理真正的调用过程。这个方法会调用 `__func__` 属性引用的原始函数，把函数的第一个参数设为绑定方法的 `__self__` 属性。这就是形参 `self` 的隐式绑定方式（终于知道class定义中`self`参数的真正含义）。

**函数会变成绑定方法，这是 Python 语言底层使用描述符的最好例证。**

两种方法可在class中设置只读属性：

1. 使用`@property`装饰器；
2. 使用实现了`__set__`和`__get__`方法的描述符类，此时，只读属性的`__set__`方法只需抛出`AttributeError`异常，并提供合适的错误消息。（这样只要出现给该属性赋值的操作，就会抛出异常）

用于验证的描述符可以只有`__set__`方法：

对于仅用于验证的描述符来说，`__set__`方法应该检查`value`参数获得的值，如果满足要求，使用描述符实例的名称为键，直接在实例的`__dict__`属性中设置。这样，从实例中读取同名属性的速度很快，因为不用经过`__get__`方法的处理。

仅有`__get__`方法的描述符可以实现高效缓存：

如果只编写了`__get__`方法，那么创建的是非覆盖型描述符。这种描述符可用于执行某些耗费资源的计算，然后为实例设置同名属性，缓存结果。同名实例属性会遮盖描述符，因此后续访问会直接从市里的`__dict__`属性中获取值，而不再触发描述符的`__get__`方法。

非特殊的方法可以被实例属性遮盖：

特殊方法不会被实例属性覆盖，例如：如果实例x中定义了一个名为`__repr__`的实例属性，那么当使用`repr(x)`显示实例x时，会不会不能得到预想中的结果？不会，因为解释器只会在类中寻找特殊方法，`repr(x)`实际执行的是`x.__class__.__repr__(x)`。

```python
class Toy:

    def __init__(self):
        setattr(self, 'b', 100)

    def func(self):
        print('call func')
```

```python
>>> toy = Toy()
>>> setattr(toy, 'a', 10)
>>> vars(toy)
{'b': 100, 'a': 10}

>>> toy.__dict__['func'] = 1000
>>> toy.func
1000

>>> toy.func()
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-95-786b489f4a79> in <module>
----> 1 toy.func()

TypeError: 'int' object is not callable
```

---

## 描述符的文档字符串和覆盖删除操作

描述符类的文档字符串用于注解托管类中的各个描述符实例。

```python
>>> help(LineItem)
Help on class LineItem in module __main__:

class LineItem(builtins.object)
 |  LineItem(description, weight, price)
 |  
 |  Methods defined here:
 |  
 |  __init__(self, description, weight, price)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |  
 |  subtotal(self)
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
 |  
 |  description
 |  
 |  price
 |  
 |  weight
```

```python
>>> help(LineItem.weight)
Help on Quantity in module __main__ object:

class Quantity(Validated)
 |  Helper class that provides a standard way to create an ABC using
 |  inheritance.
 |  
 |  Method resolution order:
 |      Quantity
 |      Validated
 |      abc.ABC
 |      AutoStorage
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  validated(self, value)
 |      return validated value or raise ValueError.
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |  
 |  __abstractmethods__ = frozenset()
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from Validated:
 |  
 |  __set__(self, instance, value)
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from Validated:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from AutoStorage:
 |  
 |  __get__(self, instance, owner)
 |  
 |  __init__(self)
 |      Initialize self.  See help(type(self)) for accurate signature.
```