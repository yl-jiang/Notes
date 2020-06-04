# 类元编程

类元编程是指在运行时创建或定制类的技艺。

在 Python 中，类是一等对象，因此任何时候都可以使用函数新建类，而无需使用 `class` 关键字。类装饰器也是函数，不过能够审查、修改，甚至把被装饰的类替换成其他类。最后，元类是类元编程最高级的工具：使用元类可以创建具有某种特质的全新类种，例如我们见过的抽象基类。

**除非开发框架，否则不要编写元类。**

下面编写一个生成类的函数工厂：

```python
def record_factory(cls_name, field_names):
    try:
        field_names = field_names.replace(',', ' ').split()
    except AttributeError:
        pass

    # field_names将作为新建类属性__slots__中的元素
    field_names = tuple(field_names)

    # 这个函数将成为新建类的__init__方法
    def __init__(self, *args, **kwargs):
        attrs = dict(zip(self.__slots__, args))
        attrs.update(kwargs)
        for name, value in attrs.items():
            setattr(self, name, value)

    # 这个函数将成为新建类的__iter__方法
    def __iter__(self):
        for name in self.__slots__:
            yield getattr(self, name)


    # 这个函数将作为新建类的__repr__方法
    def __repr__(self):
        values = ', '.join('{}={!r}'.format(*i) for i in zip(self.__slots__, self))
        return '{}({})'.format(self.__class__.__name__, values)

    cls_attrs = dict(__slots__ = field_names, 
                     __init__ = __init__, 
                     __iter__ = __iter__, 
                     __repr__ = __repr__)

    # 使用type构造方法构建新类，然后将其返回
    return type(cls_name, (object,), cls_attrs)
```

测试：

```python
# record_factory返回的是类
>>> Dog = record_factory('Dog', ['name', 'weight', 'owner'])
# Dog返回的是实例
>>> rex = Dog('Rex', 30, 'Bob')
>>> rex
Dog(name='Rex', weight=30, owner='Bob')
>>> name, weight, _ = rex
>>> name, weight
('Rex', 30)
```

通常我们把 `type` 视作函数，因为我们像函数那样使用它，例如，调用 `type(my_object)` 获取对象所属的类——作用与 `my_object.__class__` 相同。然而，`type` 是一个类。当成类使用时， 传入三个参数可以新建一个类： 

```python
MyClass = type('MyClass', (MySuperClass, MyMixin), {'x': 42, 'x2': lambda self: self.x * 2})
```

`type` 的三个参数分别是 `name`、`bases` 和 `dict`。最后一个参数是一个映射，指定新类的属性名和值。上述代码的作用与下述代码相同：

```python
class MyClass(MySuperClass, Mixin):
    x = 42

    def x2(self):
        return self.x
```

`type`本身是类，`type`的实例也是类。

---

## 定制描述符的类装饰器

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
        print('call AutoStorage.__get__')
        if instance is not None:
            print(f'call getattr(instance, {self.storage_name})')
            return getattr(instance, self.storage_name)
        else:
            return self

    def __set__(self, instance, value):
        print('call AutoStorage.__set__')
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

为第20章的LineItem的存储属性一个具有描述性的名称。

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

原来的存储属性名称：

```python
>>> LineItem.weight.storage_name
'_Quantity#0'

>>> for key, attr in LineItem.__dict__.items():
        print(f'{key:<15}', ': ', attr)
__module__      :  __main__
description     :  <__main__.NonBlank object at 0x7f3556f3a310>
weight          :  <__main__.Quantity object at 0x7f3556f3a350>
price           :  <__main__.Quantity object at 0x7f3556f3a450>
__init__        :  <function LineItem.__init__ at 0x7f3556f36710>
subtotal        :  <function LineItem.subtotal at 0x7f3556f367a0>
__dict__        :  <attribute '__dict__' of 'LineItem' objects>
__weakref__     :  <attribute '__weakref__' of 'LineItem' objects>
__doc__         :  None

>>> quan = Quantity()
>>> isinstance(quan, Validated)
True
```

我们不能使用描述性的储存属性名称，因为实例化描述符时无法得知托管属性（即绑定到描述符上的类属性）。

解决上述问题的思路是，直接对目标类进行改写并返回一个新建类。装饰器与函数装饰器非常类似，是参数为类对象的函数，返回原来的类或修改后的类。

使用装饰器方法：

```python
def entity(cls):
    print('call entity')
    for key, attr in cls.__dict__.items():
        if isinstance(attr, Validated):
            type_name = type(attr).__name__
            attr.storage_name = f'_{type_name}#{key}'
    return cls


@entity
class LineItem:

    # 我们在import LineItem或者运行时调用LineItem之前，Python解释器会运行class的顶层代码
    # 而类属性属于类的顶层代码范畴，并且装饰器发挥作用的时间也是在调用之前的，这些时间差是可以使用类装饰器的关键

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
>>> raisins = LineItem('Golden raisins', 10, 6.95)
>>> dir(raisins)[:3]
['_NonBlank#description', '_Quantity#price', '_Quantity#weight']
>>> raisins.weight
10
>>> LineItem.description.storage_name
'_NonBlank#description'
```

类装饰器有个重大缺点：只对直接依附的类有效。这意味着，被装饰的类的子类可能继承也可能不继承装饰器所做的改动，具体情况视改动的方式而定(即不具有普适性)。

元类可以定制类的层次结构。**类装饰器则不同，它只能影响一个类，而且对后代可能没有影响。**

---

## 导入时和运行时

导入时和运行时的区别——这是有效使用 Python 元编程的重要基础。

解释器会编译函数的定义体（首次导入模块时），把 函数对象绑定到对应的全局名称上，但是显然解释器不会执行函数的定义体。通常这意味着解释器在导入时定义顶层函数，但是仅当在运行时 调用函数时才会执行函数的定义体。

对类来说，情况就不同了：在导入时，解释器会执行每个类的定义体， 甚至会执行嵌套类的定义体。执行类定义体的结果是，定义了类的属性 和方法，并构建了类对象。从这个意义上理解，类的定义体属于“顶层 代码”，因为它在导入时运行。

Python中 `import` 语句，它不只是声明。在进程中首次导入模块时，还会运行所导入模块中的全部顶层代码——以后导入相同的模块则使用缓存，只做名称绑定。那些顶层代码可以做任何事，包括在”运行时“做的事，例如连接数据库。总而言之：`import`语句可以触发任何”运行时“行为(**`import` 语句会触发运行大量代码。**)。

```python
# evalsupport.py
#!/usr/bin/env python
# coding: utf-8

print('<[100]> evalsupport module start')

def deco_alpha(cls):
    print('<[200]> deco_alpha')

    def inner_1(self):
        print('<[300]> deci_alpha:inner_1')

    cls.method_y = inner_1
    return cls

class MetaAleph(type):
    print('<[400]> MetaAleph body')

    def __init__(cls, name, bases, dic):
        print('<[500]> MetaAleph.__init__')

        def inner_2(self):
            print('<[600]> MetaAleph.__init__:inner_2')

        cls.method_z = inner_2

print('<[700]> evalsupport module end')
```

```python
# evaltime.py
#!/usr/bin/env python
# coding: utf-8

from evalsupport import deco_alpha

print('<[1]> evaltime module start')

class ClassOne:

    print('<[2]> ClassOne body')

    def __init__(self):
        print('<[3]> ClassOne.__init__')

    def __del__(self):
        print('<[4]> ClassOne.__del__')

    def method_x(self):
        print('<[5]> ClassOne.method_x')

    class ClassTwo(object):
        print('<[6]> ClassTwo body')


@deco_alpha
class ClassThree:

    print('<[7]> ClassThree body')

    def method_y(self):
        print('<[8]> ClassThree.method_y')


class ClassFour(ClassThree):

    print('<[9]> ClassFour body')

    def method_y(self):
        print('<[10]> ClassFour.method_y')


if __name__ == "__main__":
    print('<[11]> ClassOne tests', 30 * '.')  # 11
    one = ClassOne()  # 3
    one.method_x()  # 5
    print('<[12]> ClassThree tests', 30 * '.')  # 12
    three = ClassThree()
    three.method_y()  # 300
    print('<[13]> ClassFour tests', 30 * '.')  # 13
    four = ClassFour()
    four.method_y()  # 10

print('<[14]> evaltime module end')  # 14
```

在Python交互式运行环境中：

```python
>>> import evaltime
<[100]> evalsupport module start
<[400]> MetaAleph body
<[700]> evalsupport module end
<[1]> evaltime module start
<[2]> ClassOne body
<[6]> ClassTwo body
<[7]> ClassThree body
<[200]> deco_alpha
<[9]> ClassFour body
<[14]> evaltime module end
```

在终端运行`evaltime.py`：

```python
python evaltime.py

<[1]> evaltime module start
<[2]> ClassOne body
<[6]> ClassTwo body
<[7]> ClassThree body
<[200]> deco_alpha
<[9]> ClassFour body
<[11]> ClassOne tests ..............................
<[3]> ClassOne.__init__
<[5]> ClassOne.method_x
<[12]> ClassThree tests ..............................
<[300]> deci_alpha:inner_1
<[13]> ClassFour tests ..............................
<[10]> ClassFour.method_y
<[14]> evaltime module end
```

虽然`ClassThree`使用了装饰器，但是其子类`ClassFour`却不受装饰器的影响，从而应证了**装饰器只能影响一个类，而且对后代可能没有影响。**

默认情况下，Python中的类是`type`类的实例，也就是说，`type`是大多数内置的类和用户定义的类的元类。

```python
# python中的内置类
>>> str.__class__
type
>>> # 用户自定义的类
LineItem.__class__
type
```

**`str`和`LineItem`不是继承自`type`，而是`str`和`LineItem`是`type`的实例。**

`object` 类和 `type` 类之间的关系很独特：`object` 是 `type` 的实例，而 `type` 是 `object` 的子类。这种关系很“神奇”，无法使用 Python 代码表述，因为定义其中一个之前另一个必须存在。

重点是：所有类都是`type`的实例，但是元类还是`type`的子类，因此可以作为制造类的工厂。具体来说，元类可以通过实现`__init__`方法可以做到类装饰器能做到的任何事情，但是作用更大。

---

## 认识元类

```python
# evaltime_meta.py

from evalsupport import deco_alpha
from evalsupport import MetaAleph

print('<[1]> evaltime_meta module start')

@deco_alpha
class ClassThree:

    print('<[2]> ClassThree body')

    def method_y(self):
        print('<[3]> ClassThree.method_y')


class ClassFour(ClassThree):

    print('<[4]> ClassFour body')

    def method_y(self):
        print('<[5]> ClassFour.method_y')

class ClassFive(metaclass=MetaAleph):

    print('<[6]> ClassFive body')

    def __init__(self):
        print('<[7]> ClassFive.__init__')

    def method_z(self):
        print('<[8]> ClassFive.method_z')

class ClassSix(ClassFive):

    print('<[9]> ClassSix body')

    def method_z(self):
        print('<[10]> ClassSix.method_z')


if __name__ == '__main__':
    # 100， 400， 700， 1， 2， 200, 4， 6， 500, 9, 500
    print('<[11]> ClassThree tests', 30 * '.')  # 11
    three = ClassThree()
    three.method_y()  # 3
    print('<[12]> ClassFour tests', 30 * '.')  # 12
    four = ClassFour()
    four.method_y()  # 5
    print('<[13]> ClassFive tests', 30 * '.')  # 13
    five = ClassFive()  # 7, 500
    five.method_z()  # 600
    print('<[14]> ClassSix tests', 30 * '.')  # 14
    six = ClassSix()
    six.method_z()  # 10

print('<[15]> evaltime_meta module end')  # 15
```

在Python交互式运行环境中：

```python
>>> import evaltime_meta
<[1]> evaltime_meta module start
<[2]> ClassThree body
<[200]> deco_alpha
<[4]> ClassFour body
<[6]> ClassFive body
<[500]> MetaAleph.__init__
<[9]> ClassSix body
<[500]> MetaAleph.__init__
<[15]> evaltime_meta module end
```

导入`evaltime_meta.py`，运行到`five = ClassFive()`时：

1. 运行`ClassFive.__init__`函数；
2. 接着交由其继承的元类`MetaAleph`继续处理：
    1. 启动`MetaAleph.__init__`方法，并将`ClassFive`作为`MetaAleph.__init__`方法的第一个参数；
    2. `MetaAleph.__init__`方法下的`inner_2`函数的`self`参数，最终指代我们在创建的类的实例，即`ClassFive`类的实例

元类的`__init__`由四个参数：`cls`, `name`, `bases`, `dic`

1. `cls`: 指代 `<class ClassFive>`；
2. `name`: 表示 `'ClassFive'`；
3. `bases`: 表示`ClassFive`继承的父类，这里`base = ()`或`base = (object,)`；
4. `dic`: 为一个字典，其中key可以为将要创建的类的属性名或者方法名，对应的value则为属性值和方法定义

`ClassSix` 类没有直接引用 `MetaAleph` 类，但是却受到了影响，因为它是 `ClassFive` 的子类，进而也是 `MetaAleph` 类的实例，所以由 `MetaAleph.__init__` 方法初始化。

总而言之：元类在导入时就开始发挥作用。

在终端运行`evaltime_meta.py`：

```python
python evaltime_meta.py

<[1]> evaltime_meta module start
<[2]> ClassThree body
<[200]> deco_alpha
<[4]> ClassFour body
<[6]> ClassFive body
<[500]> MetaAleph.__init__
<[9]> ClassSix body
<[500]> MetaAleph.__init__
<[11]> ClassThree tests ..............................
<[300]> deci_alpha:inner_1
<[12]> ClassFour tests ..............................
<[5]> ClassFour.method_y
<[13]> ClassFive tests ..............................
<[7]> ClassFive.__init__
<[600]> MetaAleph.__init__:inner_2
<[14]> ClassSix tests ..............................
<[7]> ClassFive.__init__
<[600]> MetaAleph.__init__:inner_2
<[15]> evaltime_meta module end
```

---

## 定制描述符的元类

```python
class EntityMeta(type):
    """
    元类，用于创建带有验证字段的业务实体。
    """

    def __init__(cls, name, bases, attr_dict):
        print(f'type(attr_dict): {type(attr_dict)}')
        print('EntityMeta.__init__ start')
        # 即，type(name, bases, attr_dict)
        super().__init__(name, bases, attr_dict)
        for key, attr in attr_dict.items():
            if isinstance(attr, Validated):
                print(f'key: {key}, attr: {attr}')
                # 获得描述符实例名称
                type_name = type(attr).__name__
                # 例如，修改weight的Quantity实例中的storage_name属性为'_Quantity#weight'
                attr.storage_name = f'_{type_name}#{key}'
        print('EntityMeta.__init__ end')

class Entity(metaclass=EntityMeta):
    """
    带有验证字段的业务实体。
    """
```

jupyter中将上述代码块放在一个单元格中，执行后，会显示如下结果：

```python
type(attr_dict): <class 'dict'>
EntityMeta.__init__ start
EntityMeta.__init__ end
```

```python
class LineItem(Entity):

    description = NonBlank()
    weight = Quantity()
    price = Quantity()

    def __init__(self, description, weight, price):
        print('LineItem.__init__ start')
        self.description = description
        self.weight = weight
        self.price = price
        print('LineItem.__init__ end')

    def subtotal(self):
        return self.weight * self.price
```

jupyter中将上述代码块放在一个非=单元格中，执行后，会显示如下结果：

```python
type(attr_dict): <class 'dict'>
EntityMeta.__init__ start
key: description, attr: <__main__.NonBlank object at 0x7f522bfc7fd0>
key: weight, attr: <__main__.Quantity object at 0x7f522bfc7610>
key: price, attr: <__main__.Quantity object at 0x7f522bfc7a10>
EntityMeta.__init__ end
```

测试：

```python
>>> raisins = LineItem('Golden raisins', 10, 6.95)
>>> dir(raisins)[-4:]
LineItem.__init__ start
call AutoStorage.__set__
call AutoStorage.__set__
call AutoStorage.__set__
LineItem.__init__ end
['description', 'price', 'subtotal', 'weight']

>>> raisins.price
call AutoStorage.__get__
call getattr(instance, _Quantity#price)
6.95

>>> LineItem.weight.storage_name
call AutoStorage.__get__
'_Quantity#weight'
```

---

## 元类的`__prepare__`方法

`__prepare__`方法的第一个参数是元类，随后两个参数分别是要构建的类的名称和基类组成的元组，返回值必须是映射。元类构建新类时，`__prepare__`方法返回的映射会传给`__new__`方法的最后一个参数，然后再传给`__init__`方法。`__prepare__`方法的主要作用是，对传入给元类的最后一个映射参数`dics`（`type(name, bases, dics)`）进行进一步包裹。

```python
import collections

class EntityMeta(type):

    @classmethod
    def __prepare__(cls, name, bases):
        print('call EntityMeta.__prepare__')
        # 将要构建的新类的属性映射（attr_dict）使用OrderedDict进行包裹
        return collections.OrderedDict()

    def __init__(cls, name, bases, attr_dict):
        print('call EntityMeta.__init__')
        # 此时传入的attr_dict已经是一个经过OrderedDict包裹的dict
        print('type(attr_dict): ', type(attr_dict))
        super().__init__(name, bases, attr_dict)
        cls._field_names = []

        for key, attr in attr_dict.items():
            if isinstance(attr, Validated):
                type_name = type(attr).__name__
                attr.storage_name = f'_{type_name}#{key}'
                cls._field_names.append(key)
        print('end of EntityMeta.__init__')

```

```python
class Entity(metaclass=EntityMeta):

    @classmethod
    def field_names(cls):
        for name in cls._field_names:
            yield name
```

输出：

```python
call EntityMeta.__prepare__
call EntityMeta.__init__
type(attr_dict):  <class 'collections.OrderedDict'>
end of EntityMeta.__init__
```

```python
class LineItem(Entity):

    description = NonBlank()
    weight = Quantity()
    price = Quantity()

    def __init__(self, description, weight, price):
        print('LineItem.__init__ start')
        self.description = description
        self.weight = weight
        self.price = price
        print('LineItem.__init__ end')

    def subtotal(self):
        return self.weight * self.price
```

输出：

```python
call EntityMeta.__prepare__
call EntityMeta.__init__
type(attr_dict):  <class 'collections.OrderedDict'>
end of EntityMeta.__init__
```

测试：

```python
# 注意：field_names()方法是类方法只有class才能调用，是为了方便使用class调试
>>> for name in LineItem.field_names():
        print(name)
description
weight
price

>>> LineItem._field_names
['description', 'weight', 'price']
```

`cls.__bases__`

```python
>>> LineItem.__bases__
(__main__.Entity,)
```

由类的基类组成的元组。

`cls.__qualname__`

Python 3.3 新引入的属性，其值是类或函数的限定名称，即从模块的全局作用域到类的点分路径。

```python
class ClassOne:

    print('<[2]> ClassOne body')

    def __init__(self):
        print('<[3]> ClassOne.__init__')

    def __del__(self):
        print('<[4]> ClassOne.__del__')

    def method_x(self):
        print('<[5]> ClassOne.method_x')

    class ClassTwo(object):
        print('<[6]> ClassTwo body')
```

内部类 `ClassTwo` 的 `__qualname__` 属性，其值是字符串 `'ClassOne.ClassTwo'`，而 `__name__` 属性的值是 `'ClassTwo'`。

`cls.__subclasses__()`

```python
>>> Entity.__subclasses__()
[__main__.LineItem]
```

这个方法返回一个列表，包含类的直接子类。这个方法的实现使用弱引用，防止在超类和子类（子类在 `__bases__` 属性中储存指向超类的强引用）之间出现循环引用。这个方法返回的列表中是内存里现存的子类。

`cls.mro()`

```python
>>> Entity.mro()
[__main__.Entity, object]
>>> Entity.__mro__
(__main__.Entity, object)
```

构建类时，如果需要获取储存在类属性 `__mro__` 中的超类元组， 解释器会调用这个方法。元类可以覆盖这个方法，定制要构建的类解析方法的顺序。

只有class才有`__name__`属性，class的`__class__`属性表示其继承的父类，实例没有`__name__`属性，实例的`__class__`属性表示将其实例化的class（例如，`raisins.__class__`的返回值为`__main__.LineItem`）

```python
>>> LineItem.__name__
'LineItem'
>>> raisins.__name__
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-65-c4eab770d0ec> in <module>
----> 1 raisins.__name__

AttributeError: 'LineItem' object has no attribute '__name__'
```

---

## 最最重要的事

**不要在生产代码中定义抽象基类（或元类）。**