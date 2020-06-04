# 动态属性和特性

**在 Python 中，数据的属性和处理数据的方法统称属性（attribute）。其实，方法只是可调用的属性。**

## 使用动态属性转换数据

```python
import json
from pathlib import Path

data_path = Path('../data/conference.json')

def load():
    if data_path.exists():
        with open(data_path) as fp:
            return json.load(fp)
```

测试：

```Python
>>> feed = load()
>>> sorted(feed['Schedule'].keys())
['conferences', 'events', 'speakers', 'venues']
>>> for key, value in sorted(feed['Schedule'].items()):
    print(f'{len(value):3} {key}')

  1 conferences
484 events
357 speakers
 53 venues
```

### 创建类似字典的类

即，将`feed['Schedule']['speakers'][-1]['name']`句法改写为`feed.Schedule.speakers[-1].name`访问数据。

```python
from collections import abc

class FrozenJSON:

    def __init__(self, mapping):
        self.__data = dict(mapping)

    def __getattr__(self, name):
        # 保留传入的dict作为字典类型的基本功能
        if hasattr(self.__data, name):
            return getattr(self.__data, name)
        # 使用类似于调用类属性的方式，访问字典
        else:
            if name in self.__data:
                return FrozenJSON.build(self.__data[name])
            else:
                raise AttributeError(f'{self.__class__.__name__} has no attribute {name}')

    @classmethod
    def build(cls, obj):
        # 如果访问的key所对应的value是一个Mapping对象，则继续递归使用FrozenJSON类进行再次处理
        if isinstance(obj, abc.Mapping):
            return cls(obj)
        # 如果访问的key所对应的value是一个MutableSequence对象
        elif isinstance(obj, abc.MutableSequence):
            return [cls.build(item) for item in obj]
        # 如果访问的key所对应的value不是一个容器，直接返回，这是能够使用属性方法访问数据的关键
        else:
            return obj
```

测试：

```Python
>>> raw_feed = load()
>>> feed = FrozenJSON(raw_feed)
>>> len(feed.Schedule.speakers)
357
>>> sorted(feed.Schedule.keys())
['conferences', 'events', 'speakers', 'venues']
>>> talk = feed.Schedule.events[40]
>>> type(talk)
<class '__main__.FrozenJSON'>
```

当访问无效key时，抛出`AttributeError`异常。

```python
>>> talk.flavor
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-18-abf3275fce15> in <module>
----> 1 talk.flavor

<ipython-input-13-6a8f3314d901> in __getattr__(self, name)
     15                 return FrozenJSON.build(self.__data[name])
     16             else:
---> 17                 raise AttributeError(f'{self.__class__.__name__} has no attribute {name}')
     18 
     19     @classmethod

AttributeError: FrozenJSON has no attribute flavor
```

处理访问的key与Python的保留关键字冲突的情况。

```python
from collections import abc
import keyword

class FrozenJSON:

    def __init__(self, mapping):
        self.__data = {}
        for key, value in mapping.items():
            # 当key与Python的保留关键字冲突时，进行替换
            if keyword.iskeyword(key):
                key += '_'
            self.__data[key] = value

    def __getattr__(self, name):
        # 保留传入的dict作为字典类型的基本功能
        if hasattr(self.__data, name):
            return getattr(self.__data, name)
        # 使用类似于调用类属性的方式，访问字典
        else:
            if name in self.__data:
                return FrozenJSON.build(self.__data[name])
            else:
                raise AttributeError(f'{self.__class__.__name__} has no attribute {name}')

    @classmethod
    def build(cls, obj):
        # 如果访问的key所对应的value是一个Mapping【单层容器】对象，则继续递归使用FrozenJSON类进行再次处理
        if isinstance(obj, abc.Mapping):
            return cls(obj)
        # 如果访问的key所对应的value是一个MutableSequence对象【多层嵌套容器】
        elif isinstance(obj, abc.MutableSequence):
            return [cls.build(item) for item in obj]
        # 如果访问的key所对应的value不是一个容器，直接返回
        else:
            return obj
```

测试：

```python
>>> grad = FrozenJSON({'name': 'Jim Bo', 'class': 1982}) 
>>> grad.class_
1982
```

FrozenJSON 类的关键是 `__getattr__` 方法。我们要记住重要的一点，仅当无法使用常规的方式获取属性（即在实例、类或超类中找不到指定的属性），解释器才会调用特殊的 `__getattr__` 方法。

---

## `__new__`

`__new__`方法是个特殊的类方法（使用特殊方式处理，使用时不必添加`@classmethod`装饰器），一般情况下，该方法返回一个实例。返回的实例会传给`__init__`，并作为`__init__`的第一个参数（即`self`）。

因为调用 `__init__` 方法时要传入实例，而且禁止返回任何值，所以 `__init__` 方法其实是“初始化方法”。真正的构造方法是 `__new__`。 我们几乎不需要自己编写 `__new__` 方法，因为从 `object` 类继承的实现已经足够了。 

上述说明的过程，即`__new__` 方法到 `__init__` 方法，是最常见的，但不是唯一的。`__new__` 方法也可以返回其他类的实例，此时，解释器不会调用 `__init__` 方法。

使用`__new__`代替`build`函数重构`FrozenJSON`：

```python
from collections import abc
import keyword

class FrozenJSON:
    
    def __new__(cls, arg):
        if isinstance(arg, abc.Mapping):
            # 返回FrozenJSON实例，调用__init__方法，返回的实例作为__init__方法的第一个参数
            return super().__new__(cls)
        elif isinstance(arg, abc.MutableSequence):
            # 同样返回FrozenJSON实例，会调用__init__方法
            return [cls(item) for item in arg]
        else:
            # __new__方法返回其他类的实例，因此不会调用__init__方法
            return arg

    def __init__(self, mapping):
        self.__data = {}
        for key, value in mapping.items():
            if keyword.iskeyword(key):
                key += '_'
            self.__data[key] = value

    def __getattr__(self, name):
        # 保留传入的dict作为字典类型的基本功能
        if hasattr(self.__data, name):
            return getattr(self.__data, name)
        # 使用类似于调用类属性的方式，访问字典
        else:
            if name in self.__data:
                # 现在只需调用 FrozenJSON 的构造方法
                return FrozenJSON(self.__data[name])
            else:
                raise AttributeError(f'{self.__class__.__name__} has no attribute {name}')
```

`__new__` 方法的第一个参数是类，因为创建的对象通常是那个类的实例。所以，在 `FrozenJSON.__new__` 方法中，`super().__new__(cls)` 表达式会调用 `object.__new__(FrozenJSON)`，而 `object` 类构建的实例其实是 `FrozenJSON` 实例，即那个实例的 `__class__` 属性存储的是 `FrozenJSON` 类的引用。

### 使用`__new__`实现单例模式

```python
class Singleton:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance
```

测试：

```python
>>> s1 = Singleton()
>>> s2 = Singleton()
>>> print(s1)
<__main__.Singleton object at 0x7f56a7289d50>
>>> print(s2)
<__main__.Singleton object at 0x7f56a7289d50>
```

### 使用`__new__`实现工厂模式

```python
class Fruit(object):
    def __init__(self, name):
        self.name = name

    def print_color(self):
        print(f"{self.name} is in red")

class Apple(Fruit):
    def __init__(self):
        pass

    def print_color(self):
        print("apple is in red")

class Orange(Fruit):
    def __init__(self):
        pass

    def print_color(self):
        print("orange is in orange")

class FruitFactory(object):
    fruits = {"apple": Apple, "orange": Orange}

    def __new__(cls, name):
        if name in cls.fruits.keys():
            return cls.fruits[name]()
        else:
            return Fruit(name)
```

测试：

```python
>>> fruit1 = FruitFactory("apple")
>>> fruit2 = FruitFactory("orange")
>>> fruit3 = FruitFactory("some")
>>> fruit1.print_color()
apple is in red
>>> fruit2.print_color()
orange is in orange
>>> fruit3.print_color()
some is in red
```

---

## 动态属性

上述所有的在用户看来是使用点号访问属性值的方法，其实在内部最终都是由`'return arg'`得到的，并不是真正意义上的访问这些属性。

```python
data_path = Path('../data/conference.json')

def load():
    if data_path.exists():
        with open(data_path) as fp:
            return json.load(fp)

import warnings

DB_NAME = '../data/schedule1_db'
CONFERENCE = 'conference.115'

class Record:

    def __init__(self, **kwargs):
        # 关键代码：使用关键字参数传入的属性构建实例！！！
        self.__dict__.update(kwargs)

def load_db(db):
    raw_data = load()
    warnings.warn('loading ' + DB_NAME)
    for collection, rec_list in raw_data['Schedule'].items():
        record_type = collection[:-1]
        for record in rec_list:
            # 修改record中key值为'serial'对应的value
            key = f"{record_type}.{record['serial']}"
            record['serial'] = key
            db[key] = Record(**record)
```

对象的 `__dict__` 属性中存储着对象的属性——前提是类中没有声明 `__slots__` 属性，因此，更新实例的 `__dict__` 属性， 把值设为一个映射，能快速地在那个实例中创建一堆属性。

测试：

```python
>>> import shelve
>>> db = shelve.open(DB_NAME)
>>> if CONFERENCE not in db:
        load_db(db)
/home/dk/anaconda3/envs/fun/lib/python3.7/site-packages/ipykernel_launcher.py:21: UserWarning: loading ../data/schedule1_db
>>> speaker = db['speaker.3471']
>>> type(speaker)
<class '__main__.Record'>
>>> speaker.name, speaker.twitter
'Anna Martelli Ravenscroft', 'annaraven'
>>> db.close()
```

---

## 使用特性获取链接的记录

```python
import warnings
import inspect

DB_NAME = '../data/schedule2_db'
CONFERENCE = 'conference.115'


class Record:

    def __init__(self, **kwargs):
        # 关键代码：使用关键字参数传入的属性构建实例！！！
        self.__dict__.update(kwargs)

    def __eq__(self, other):
        """
        该方法只为方便调试。
        """
        if isinstance(other, Record):
            return self.__dict__ == other.__dict__
        else:
            return NotImplement
```

```python
class MissingDatabaseError(RuntimeError):
    """
    需要数据库但没有指定数据库时抛出
    """

class DbRecord(Record):
    """
    实现有关数据库的操作。
    """

    __db = None

    @staticmethod
    def set_db(db):
        DbRecord.__db = db

    @staticmethod
    def get_db():
        return DbRecord.__db

    @classmethod
    def fetch(cls, ident):
        print('call fetch of DbRecord class ...')
        db = cls.get_db()
        try:
            return db[ident]
        except TypeError:
            if db is None:
                msg = "database not set; call '{}.set_db(my_db)'"
                raise MissingDatabaseError(msg.format(cls.__name__))
            else:
                raise

    def __repr__(self):
        """
        如果记录有 serial 属性，在字符串表示形式中使用。
        否则，调用继承的 __repr__ 方法。
        """
        if hasattr(self, 'serial'):
            cls_name = self.__class__.__name__
            return f'<{cls_name} serial = {self.serial}>'
        else:
            return super().__repr__()

class Event(DbRecord):

    @property
    def venue(self):
        print('call venue ...')
        key = f'venue.{self.venue_serial}'
        # 显式调用继承自父类的fetch方法，可以已使用self.fetch()，但是这样做更安全
        return self.__class__.fetch(key)

    @property
    def speakers(self):
        print('call speakers ...')
        if not hasattr(self, '_speaker_objs'):
            spkr_serials = self.__dict__['speakers']
            fetch = self.__class__.fetch
            self._speaker_objs = [fetch(f'speaker.{key}') for key in spkr_serials]
        return self._speaker_objs

    def __repr__(self):
        if hasattr(self, 'name'):
            cls_name = self.__class__.__name__
            return '<{} {!r}>'.format(cls_name, self.name)
        else:
            return super().__repr__()

def load_db(db):
    raw_data = load()
    warnings.warn('loading ' + DB_NAME)
    for collection, rec_list in raw_data['Schedule'].items():
        record_type = collection[:-1]
        cls_name = record_type.capitalize()
        # 从模块的全局作用域中获取对应名称的对象；如果找不到，则使用 DbRecord。
        cls = globals().get(cls_name, DbRecord)

        if inspect.isclass(cls) and issubclass(cls, DbRecord):
            # factory是Event类
            factory = cls
        else:
            # factory是DbRecord类
            factory = DbRecord

        for record in rec_list:
            key = '{}.{}'.format(record_type, record['serial'])
            record['serial'] = key
            print(f'key : {key} -> dict: {record}')
            print('-' * 50)
            db[key] = factory(**record)
        print('#' * 50)
```

自定义的异常通常是标志类，没有定义体。写一个文档字符串，说明异常的用途，比只写一个 `pass` 语句要好。

`globals()` 函数会以字典类型返回当前位置的全部全局变量。

本例中构建动态属性的代码逻辑：

1. 构造`key`，确保所有`key`都是全局唯一的；
2. 每个`key`通过`load_db()`函数，被实例化为`DbRecord`或者`Event`对象；
3. 将每个`key`对应的`value`（在本例中为一个dict）通过`Record`的`self.__dict__.update(kwargs)`方法，为`key`对应的实例对象通过传入关键字参数的方式为其创建动态属性。

测试：

```python
>>> DbRecord.set_db(db)
>>> event = DbRecord.fetch('event.33457')
>>> event
call fetch of DbRecord class ...
<Event 'Refactoring 101'>
```

```python
>>> event.venue.name
call venue ...
call fetch of DbRecord class ..
```

```python
>>> db['event.33457'].name
'Refactoring 101'
>>> event.venue
call venue ...
call fetch of DbRecord class ...
<DbRecord serial = venue.1449>
```

```python
>>> event.speakers
call speakers ...
call fetch of DbRecord class ...
[<DbRecord serial = speaker.169862>]
```

---

## 使用特性验证属性

类的特性能影响实例属性的寻找方式。

**特性会覆盖实例属性。**

特性都是类属性，但是特性管理的其实是实例属性的存取。

```python
class LineItem:

    def __init__(self, description, weight, price):
        self.description = description
        self.weight = weight
        self.price = price

    def subtotal(self):
        return self.weight * self.price

    @property
    def weight(self):
        print('call @property')
        return self.__weight

    @weight.setter
    def weight(self, value):
        print('call @weight.setter')
        if value > 0:
            self.__weight = value
        else:
            raise ValueError('Value must be > 0')
```

测试：

```python
>>> walnuts = LineItem('walbuts', 100, 10)
call @weight.setter
>>> walnuts.weight
call @property
100
```

给`weight`赋值时会直接调用`@weight.setter`修饰的方法，且该方法的名称必须为`weight`。访问属性`weight`时，并不像以往那样直接从`__init__`中读取，而是会直接调用`@property`修饰的名称为`weight`的方法。

```python
class Toy:

    def __init__(self):
        self.a = 10

    @property
    def a(self):
        return 20
```

测试：

```python
>>> toy = Toy()
>>> toy.a
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-22-213dcc7a0d7b> in <module>
----> 1 toy = Toy()
      2 toy.a

<ipython-input-21-f0823c73e358> in __init__(self)
      2 
      3     def __init__(self):
----> 4         self.a = 10
      5 
      6     @property

AttributeError: can't set attribute
```

实例化Toy，当初始化函数`__init__()`运行到`self.a = 10`时，由于Python检查到了有名称为`a`的特性，此时`self.a`不再是一般的实例属性而是类特性，语句`self.a = 10`代表了给特性赋值，因此此时Toy中必须有`@a.setter`修饰的名称为`a`的方法，否者抛出异常。

```python
class Toy:

    def __init__(self):
        self.a = 10

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        self._a = value
```

```python
>>> toy = Toy()
>>> toy.a
10
```

初始化函数`__init__()`运行到`self.a = 10`时，会调用`@a.setter`装饰器修饰的方法，即，`self.a = 10`等价于`a(toy, 10)`。

**注意**， 上述代码不能改写为：

```python
class Toy:

    def __init__(self):
        self.a = 10

    @property
    def a(self):
        return self.a

    @a.setter
    def a(self, value):
        self.a = value
```

因为这样会无限递归调用`@a.setter`装饰器修饰的方法。

### 实例属性覆盖类的数据属性

```python
class Class:

    # 类的数据属性
    data = 'the class data attr'

    # 类的特性
    @property
    def prop(self):
        return 'the prop value'
```

测试：

```python
>>> obj = Class()
# vars()返回实例属性
>>> print(vars(obj))
{}
>>> obj.data = 'bar'
>>> vars(obj)
{'data': 'bar'}
>>> Class.data
the class data attr
```

### 实例属性不会覆盖类特性

```python
>>> Class.prop
<property at 0x7f56c81e3470>
>>> obj.prop
'the prop value'
# 尝试覆盖特性
>>> obj.prop = 'foo'
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-53-950b619b2a59> in <module>
----> 1 obj.prop = 'foo'

AttributeError: can't set attribute
```

实例的`__dict__`保存了所有实例属性，即使强行将`prop`写入`__dict__`，依然不能覆盖同名的类特性。

```python
>>> obj.__dict__['prop'] = 'foo'
>>> vars(obj)
{'data': 'bar', 'prop': 'foo'}
>>> obj.prop
'the prop value'
```

```python
# 销毁类特性
>>> Class.prop = 'destroy'
>>> obj.prop
'foo'
```

只有在销毁了类特性`prop`后，实例属性`prop`才暴露出来。

### 动态添加类特性

虽然内置的 `property` 经常用作装饰器，但它其实是一个类。在 Python 中，函数和类通常可以互换，因为二者都是可调用的对象，而且没有实例化对象的 `new` 运算符，所以调用构造方法与调用工厂函数没有区别。 此外，只要能返回新的可调用对象，代替被装饰的函数，二者都可以用作装饰器。 `property` 构造方法的完整签名如下： 

```python
property(fget=None, fset=None, fdel=None, doc=None)
```

所有参数都是可选的，如果没有把函数传给某个参数，那么得到的特性对象就不允许执行相应的操作。

不使用装饰器定义特性的“经典”句法:

```python 
class LineItem:

    def __init__(self, description, weight, price):
        self.description = description
        self.weight = weight
        self.price = price

    def subtotal(self):
        return self.weight * self.price

    def get_weight(self):
        return self.__weight

    def set_weight(self, value):
        if value > 0:
            self.__weight = value
        else:raise ValueError('value must be > 0')

    weight = property(get_weight, set_weight)
```

使用`property`为上述简单的栗子动态的添加特性：

```python
>>> obj.data
'bar'
>>> Class.data
'the class data attr'
# 为Class动态的添加特性
>>> Class.data = property(lambda self: 'the data prop value')
# 此时data已经变成类的特性，而不再是类的数据属性，因此会覆盖obj的实例属性data
>>> obj.data
'the data prop value'
# 销毁类特性'data'
>>> del Class.data
# 实例属性'data'重新暴露出来
>>> obj.data
'bar'
```

总而言之，`obj.attr`这样的表达式不会从`obj`（实例）开始寻找`attr`，而是从`obj.__class__`开始，而且，仅当类中没有名为`attr`的特性时，Python才会在`obj`实例中寻找。

### 为特性添加文档字符串说明

```python
class Foo:

    @property
    def bar(self):
        """
        The bar attribute.
        """
        return self.__dict__['bar']

    @bar.setter
    def bar(self, value):
        """
        在这里添加的文档字符串说明，不会在help(Foo)中显示。
        """
        self.__dict__['bar'] = value
```

```python
>>> foo = Foo()
>>> foo.bar = 10
>>> help(foo)
Help on Foo in module __main__ object:

class Foo(builtins.object)
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
 |  
 |  bar
 |      The bar attribute.
```

---

## 特性工厂函数

如果我们想在实例化`LineItem`时，检查传入的参数是否符合要求（都必须大于0），可以这样做：

```python
class LineItem:

    def __init__(self, description, weight, price):
        self.description = description
        self.weight = weight
        self.price = price

    def subtotal(self):
        return self.weight * self.price

    @property
    def weight(self):
        print('call @property weight')
        return self.__weight

    @weight.setter
    def weight(self, value):
        print('call @weight.setter')
        if value > 0:
            self.__weight = value
        else:
            raise ValueError('Value must be > 0')

    @property
    def price(self):
        print('call @property price')
        return self.__weight

    @price.setter
    def price(self, value):
        print('call @price.setter')
        if value > 0:
            self.__weight = value
        else:
            raise ValueError('Value must be > 0')
```

测试：

```python
>>> li = LineItem('somestuff', 10, 20)
call @weight.setter
call @price.setter
>>> li2 = LineItem('somestuff2', 10, -30)
call @weight.setter
call @price.setter
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-77-d4e8c8bf83cc> in <module>
----> 1 li2 = LineItem('somestuff2', 10, -30)

<ipython-input-74-d48259f23d0f> in __init__(self, description, weight, price)
      4         self.description = description
      5         self.weight = weight
----> 6         self.price = price
      7 
      8     def subtotal(self):

<ipython-input-74-d48259f23d0f> in price(self, value)
     33             self.__weight = value
     34         else:
---> 35             raise ValueError('Value must be > 0')

ValueError: Value must be > 0
```

这样做成功的实现了我们想要的，但是，唯一的缺陷是我们对属性`weight`和`price`的进行了同样的操作，但是不得不重复书写代码，如果还有更多类似的属性需要这样做，那么上述的代码实现就显得没有效率。

使用特性工厂函数实现上述逻辑：

```python
def quantity(storage_name):

    def qty_getter(instance):
        return instance.__dict__[storage_name]

    def qty_setter(instance, value):
        if value > 0:
            instance.__dict__[storage_name] = value
        else:
            raise ValueError('value must be > 0')

    # 这是Python2中@property的用法，也是更具一般性的使用装饰器构建特性的方法
    return property(fget=qty_getter, fset=qty_setter)
```

```python

class LineItem:

    # 使用特性工厂函数，构建类特性
    weight = quantity('weight')
    price = quantity('price')

    def __init__(self, description, weight, price):
        self.description = description
        # 此时weight已经是类特性，这里表示给类特性赋值
        self.weight = weight
        self.price = price

    def subtotal(self):
        # 获取类特性的值完成计算
        return self.weight * self.price
```

LineItem 类没有干扰人的读值方法和设值方法，看起来舒服多了。 

测试：

```Python
>>> nutmeg = LineItem('Moluccan nutmeg', 8, 13.95)
# 通过特性读取weight和price的值
>>> nutmeg.weight, nutmeg.price
(8, 13.95)
>>> nutmeg.__dict__['weight'] = -100
>>> nutmeg.weight, nutmeg.price
(-100, 13.95)
>>> sorted(vars(nutmeg).items())
[('description', 'Moluccan nutmeg'), ('price', 13.95), ('weight', 100)]
```

工厂函数构建的特性利用了特性的行为：`weight` 特性覆盖了 `weight` 实例属性，因此对 `self.weight` 或 `nutmeg.weight` 的每个引用都由特性函数处理，只有直接存取 `__dict__` 属性才能跳过特 性的处理逻辑。

---

## 处理属性删除操作

使用 Python 编程时不常删除属性，通过特性删除属性更少见。但是，Python 支持这么做。

```python
class BlackKnight:

    def __init__(self):
        self.members = ['an arm', 'another arm', 'an leg', 'another leg']
        self.phrases = ["Tis but a scratch", 
                       "It's just a flesh wound", 
                       "I'm invincible", 
                       "All right, we'll call it a draw."]

    @property
    def member(self):
        print('next member is :')
        return self.members[0]

    @member.deleter
    def member(self):
        text = 'BLACK KNIGHT (loses {}) \n-- {}'
        print(text.format(self.members.pop(0), self.phrases.pop(0)))
```

测试：

```python
>>> knight = BlackKnight()
>>> knight.member
next member is :
'an arm'
>>> del knight.member
BLACK KNIGHT (loses an arm)
-- Tis but a scratch
>>> del knight.member
BLACK KNIGHT (loses another arm)
-- It's just a flesh wound
>>> del knight.member
BLACK KNIGHT (loses an leg)
-- I'm invincible
>>> del knight.member
BLACK KNIGHT (loses another leg)
-- All right, we'll call it a draw.
```

在不使用装饰器的经典调用句法中，`fdel` 参数用于设置删值函数.

```python
class BlackKnight:

    def __init__(self):
        self.members = ['an arm', 'another arm', 'an leg', 'another leg']
        self.phrases = ["Tis but a scratch", 
                       "It's just a flesh wound", 
                       "I'm invincible", 
                       "All right, we'll call it a draw."]

    def member_getter(self):
        print('next member is :')
        return self.members[0]

    def member_deleter(self):
        text = 'BLACK KNIGHT (loses {}) \n-- {}'
        print(text.format(self.members.pop(0), self.phrases.pop(0)))

    member = property(fget=member_getter, fdel=member_deleter)
```

测试：

```Python
>>> knight = BlackKnight()
>>> knight.member
next member is :
'an arm'
>>> del knight.member
BLACK KNIGHT (loses an arm)
-- Tis but a scratch
>>> del knight.member
BLACK KNIGHT (loses another arm)
-- It's just a flesh wound
>>> del knight.member
BLACK KNIGHT (loses an leg)
-- I'm invincible
>>> del knight.member
BLACK KNIGHT (loses another leg)
-- All right, we'll call it a draw.
```

---

## 影响属性处理方式的特殊属性 

`__class__`

对象所属类的引用（即 `obj.__class__` 与 `type(obj)` 的作用相 同）。Python 的某些特殊方法，例如 `__getattr__`，只在对象的类中寻找，而不在实例中寻找。

`__dict__`

一个映射，存储对象或类的可写属性。**有 `__dict__` 属性的对象， 任何时候都能随意设置新属性**。如果类有 `__slots__` 属性，它的实例可能没有 `__dict__` 属性。

`__slots__`

类可以定义这个这属性，限制实例只能有哪些属性。`__slots__` 属性的值是一个字符串组成的元组，指明允许有的属性。 如果 `__slots__` 中没有 `'__dict__'`，那么该类的实例就没有 `__dict__` 属性，实例只允许有指定名称的属性。 

---

## 处理属性的内置函数

`dir([object])`

列出对象的大多数属性。[官方文档](https://docs.python.org/3/library/functions.html#dir)说，`dir` 函数的目的是交互式使用，因此没有提供完整的属性列表，只列出一组“重要的”属性名。`dir` 函数能审查有或没有 `__dict__` 属性的对象。`dir` 函数不会列出 `__dict__` 属性本身，但会列出其中的键。`dir` 函数也不会列出类的几个特殊属性，例如 `__mro__`、`__bases__` 和 `__name__`。**如果没有指定可选的 `object` 参数，`dir` 函数会列出当前作用域中的名称。**

`getattr(object, name[, default])`

从 `object` 对象中获取 `name` 字符串对应的属性。获取的属性可能来自对象所属的类或超类。如果没有指定的属性，`getattr` 函数抛出 `AttributeError` 异常，或者返回 `default` 参数的值（如果设定了这个参数的话）。 

`hasattr(object, name)`

如果 `object` 对象中存在指定的属性，或者能以某种方式（例如继承）通过 `object` 对象获取指定的属性，返回 `True`。[文档](https://docs.python.org/3/library/functions.html#hasattr)说道：“这个函数的实现方法是调用 `getattr(object, name)` 函数，看看是否抛出 `AttributeError` 异常。”

`setattr(object, name, value)`

把 `object` 对象指定属性的值设为 `value`，前提是 `object` 对象能接受那个值。这个函数可能会创建一个新属性，或者覆盖现有的属性。(`setarrt(instance, 'somename', somevalue)`等同于`instance.__dict__['somename'] = somevalue`)

`vars([object])`

返回 `object` 对象的 `__dict__` 属性；如果实例所属的类定义了 `__slots__` 属性，实例没有 `__dict__` 属性，那么 `vars` 函数不能处理那个实例（相反，`dir` 函数能处理这样的实例）。**如果没有指定参数， 那么 `vars()` 函数的作用与 `locals()` 函数一样：返回表示本地作用域的字典。**

---

## 处理属性的特殊方法

使用点号或内置的 `getattr`、`hasattr` 和 `setattr` 函数存取属性都会触发相应的特殊方法。但是，直接通过实例的 `__dict__` 属性读写属性不会触发这些特殊方法——如果需要，通常会使用这种方式跳过特殊方法。 

`__delattr__(self, name)`

只要使用 `del` 语句删除属性，就会调用这个方法。例如，`del obj.attr` 语句触发 `Class.__delattr__(obj, 'attr')` 方法。

`__dir__(self)`

把对象传给 `dir` 函数时调用，列出属性。例如，`dir(obj)` 触发 `Class.__dir__(obj)` 方法。 

`__getattr__(self, name)`

表达式 `obj.no_such_attr`、`getattr(obj, 'no_such_attr')` 和 `hasattr(obj, 'no_such_attr')` 可能会触发 `Class.__getattr__(obj, 'no_such_attr')` 方法，但是，仅当在 `obj`、`Class` 和超类中找不到指定的属性时才会触发。

`__getattribute__(self, name)`

尝试获取指定的属性时总会调用这个方法，不过，寻找的属性是特殊属性或特殊方法时除外。点号与 `getattr` 和 `hasattr` 内置函数会触发这个方法。调用 `__getattribute__` 方法且抛出 `AttributeError` 异常时，才会调用 `__getattr__` 方法。为了在获取 `obj` 实例的属性时不导致无限递归，`__getattribute__` 方法的实现要使用 `super().__getattribute__(obj, name)`。 

`__setattr__(self, name, value)`

尝试设置指定的属性时总会调用这个方法。点号和 `setattr` 内置函数会触发这个方法。例如，`obj.attr = 42` 和 `setattr(obj, 'attr', 42)` 都会触发 `Class.__setattr__(obj, ‘attr’, 42)` 方法。

特殊方法 `__getattribute__` 和 `__setattr__` 不管怎样都会调用，几乎会影响每一次属性存取，因此比 `__getattr__` 方法（**只处理不存在的属性名**）更难正确使用。与定义这些特殊方法相比，使用特性或描述符相对不易出错。 