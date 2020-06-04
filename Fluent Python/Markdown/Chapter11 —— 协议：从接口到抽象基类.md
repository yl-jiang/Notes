

# Python中的接口和协议

对Python来说，“X类对象”、“X协议”和“X接口”都是一个意思。

```python
class FrenchDeck3:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = "spades diamonds clubs hearts".split()

    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits
                      for rank in self.ranks]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]

    def __setitem__(self, position, value):
        self._cards[position] = value

    def __delitem__(self, position):
        del self._cards[position]

    def insert(self, position, value):
        self._cards.insert(position, value)
```

测试：

虽然没有实现`__iter__`方法，但是下面的程序依然可以按照预期运行。

```python
card2 = FrenchDeck3()

for c in card2:
    print(c)
```

output：

```shell
Card(rank='2', suit='spades')
Card(rank='3', suit='spades')
Card(rank='4', suit='spades')
Card(rank='5', suit='spades')
Card(rank='6', suit='spades')
Card(rank='7', suit='spades')
Card(rank='8', suit='spades')
Card(rank='9', suit='spades')
Card(rank='10', suit='spades')
Card(rank='J', suit='spades')
Card(rank='Q', suit='spades')
Card(rank='K', suit='spades')
Card(rank='A', suit='spades')
Card(rank='2', suit='diamonds')
Card(rank='3', suit='diamonds')
Card(rank='4', suit='diamonds')
Card(rank='5', suit='diamonds')
Card(rank='6', suit='diamonds')
Card(rank='7', suit='diamonds')
Card(rank='8', suit='diamonds')
Card(rank='9', suit='diamonds')
Card(rank='10', suit='diamonds')
Card(rank='J', suit='diamonds')
Card(rank='Q', suit='diamonds')
Card(rank='K', suit='diamonds')
Card(rank='A', suit='diamonds')
Card(rank='2', suit='clubs')
Card(rank='3', suit='clubs')
Card(rank='4', suit='clubs')
Card(rank='5', suit='clubs')
Card(rank='6', suit='clubs')
Card(rank='7', suit='clubs')
Card(rank='8', suit='clubs')
Card(rank='9', suit='clubs')
Card(rank='10', suit='clubs')
Card(rank='J', suit='clubs')
Card(rank='Q', suit='clubs')
Card(rank='K', suit='clubs')
Card(rank='A', suit='clubs')
Card(rank='2', suit='hearts')
Card(rank='3', suit='hearts')
Card(rank='4', suit='hearts')
Card(rank='5', suit='hearts')
Card(rank='6', suit='hearts')
Card(rank='7', suit='hearts')
Card(rank='8', suit='hearts')
Card(rank='9', suit='hearts')
Card(rank='10', suit='hearts')
Card(rank='J', suit='hearts')
Card(rank='Q', suit='hearts')
Card(rank='K', suit='hearts')
Card(rank='A', suit='hearts')
```

虽然没有实现`__contains__`方法，但是`in`语句依然可以按照预期执行。

```python
Card(rank='2', suit='spades') in card2
```

输出：

```shell
True
```

一句话，**Python会特殊看待看起来像是序列的对象。**

虽然没有 `__iter__` 方法，但是 `FrenchDeck3` 实例是可迭代的对象，因为发现有 `__getitem__` 方法时，Python 会调用它，传入从 0 开始的整数索引， 尝试迭代对象（这是一种后备机制）。尽管没有实现 `__contains__` 方法，但是 Python 足够智能，能迭代 `FrenchDeck3` 实例，因此也能使用 `in` 运算符：Python 会做全面检查，看看有没有指定的元素。 综上，鉴于序列协议的重要性，如果没有`__iter__`和 `__contains__` 方法，Python 会调用 `__getitem__` 方法，设法让迭代和 `in` 运算符可用。

---

## 猴子补丁

在`class`的定义之外，动态的为`class`增加方法或属性。

```python
class Monkey:

    def __init__(self):
        self.a = 5

    def func1(self):
        print('it is func1')

    def func2(self):
        print('it is func2')

def func3(self):
    print(self.a)
    print('it is monkey patch')
```

测试：

```python
m = Monkey()
m.func1()

# 为Monkey增加补丁
Monkey.func3 = func3
m.func3()
```

输出：

```python
it is func1
5
it is monkey patch
```

---

## 抽象基类

标准库提供了大量的抽象基类，大多数的抽象基类在`collections.abc`模块，不过其他地方也有一些，例如， `numbers`和`io`模块中有一些抽象基类。但是，`collections.abc`中的抽象基类最常用。

| ABC | Inherits from | Abstract Methods | Mixin Methods(具体方法) |
| ------ | ------ | ------ | ------ |
| Container |  | `__contains__` |  |
| Hashable |  | `__hash__` |  |
| Iterable |  | `__iter__` |  |
| Iterator | Iterable | ` __next__` | `__iter__` |
| Reversible | Iterable | `__reversed__` |  |
| Generator | Iterable | send, throw | close, `__iter__`, `__next__` |
| Sized |  | `__len__` |  |
| Callable |  | `__call__` |  |
| Collection | Sized, Iterable, Container | `__contains__`, `__iter__`, `__len__` | |
| Sequence | Reversible, Collection | `__getitem__`, `__len__` | `__contains__`, `__iter__`, `__reversed__`, index, and count |
| MutableSequence | Sequence | `__getitem__`, `__setitem__`,` __delitem__`, `__len__`, insert | Inherited Sequence methods and append, reverse, extend, pop, remove, and `__iadd__` |
| ByteString | Sequence | `__getitem__`, `__len__` | Inherited Sequence methods |
| Set | Collection | `__contains__`, `__iter__`, `__len__` | `__le__`, `__lt__`, `__eq__`, `__ne__`, `__gt__`, `__ge__`, `__and__`, `__or__`, `__sub__`, `__xor__`, and isdisjoint |
| MutableSet | Set | `__contains__`, `__iter__`, `__len__`, add, discard | Inherited Set methods and clear, pop, remove, `__ior__`, `__iand__`, `__ixor__`, and `__isub__` |
| Mapping | Collection | `__getitem__`, `__iter__`, `__len__` | `__contains__`, keys, items, values, get, `__eq__`, and `__ne__` |
| MutableMapping | Mapping | `__getitem__`, `__setitem__`, `__delitem__`, `__iter__`, `__len__` | Inherited Mapping methods and pop, popitem, clear, update, and setdefault |
| MappingView | Sized |  | `__len__` |
| ItemsView | MappingView, Set |  | `__contains__`, `__iter__` |
| KeysView | MappingView, Set |  | `__contains__`, `__iter__` |
| ValuesView | MappingView, Set |  | `__contains__`, `__iter__` |
| Awaitable |  | `__await__` |  |
| Coroutine | Awaitable | send, throw | close |
| AsyncIterable |  | `__aiter__` |  |
| AsyncIterator | AsyncIterable | `__anext__` | `__aiter__` |
| AsyncGenerator | AsyncIterator | asend, athrow | aclose, `__aiter__`, `__anext__` |
|  |  |  |  |

[Reference](https://docs.python.org/3/library/collections.abc.html#collections-abstract-base-classes)

### Iterable、Container 和 Sized

各个集合应该继承这三个抽象基类，或者至少实现兼容的协议。`Iterable` 通过 `__iter__` 方法支持迭代，`Container` 通过 `__contains__` 方法支持 `in`运算符，`Sized` 通过 `__len__` 方法支持 `len()` 函数。

### Sequence、Mapping 和 Set

这三个是主要的不可变集合类型，而且各自都有可变的子类。

### MappingView

在 Python 3 中，映射方法 `.items()`、`.keys()` 和 `.values()` 返回的对象分别是 `ItemsView`、`KeysView` 和 `ValuesView` 的实例。前两个类还从 `Set` 类继承了丰富的接口。

### Callable 和 Hashable

这两个抽象基类与集合没有太大的关系，只不过因为 `collections.abc` 是标准库中定义抽象基类的第一个模块，而它们又太重要了，因此才把它们放到 `collections.abc` 模块中。我从未见过 `Callable` 或 `Hashable` 的子类。这两个抽象基类的主要作用是为内置函数 `isinstance` 提供支持，以一种安全的方式判断对象能不能调用或散列。

测试对象是否是可调用的，使用`'callable(obj)'`；测试对象是否是可散列的，使用`'isinstance(obj, Hashable)'`

### Iterator

它是 `Iterable` 的子类。

---

## 自定义并使用抽象基类

+ **抽象基类中也可以包含具体方法，抽象基类中的具体方法只能依赖抽象基类中定义的接口（即只能使用抽象基类中的其它具体方法、抽象方法或特性）**；
+ **抽象基类中抽象方法也可以有具体的实现代码，但是即便是实现了，子类中也必须覆盖该方法，子类中可以调用`super()`方法为它添加功能，而不是从头实现**；
+ **抽象方法只存在于抽象基类中**；
+ **抽象基类不可以实例化**；
+ **继承抽象基类的子类，必须覆盖抽象基类中的所有抽象方法，否则，它也将被当作另一种抽象基类【类似C++中包含纯虚函数的class】**；
+ **继承抽象基类的子类，可以选择性覆盖抽象基类中的具体方法**

定义抽象基类：

```python
import abc

class Tombola(abc.ABC):

    # 定义抽象方法
    @abc.abstractmethod
    def load(self, iterable):
        """"""

    # 定义抽象方法
    @abc.abstractmethod
    def pick(self):
        """"""

    # 抽象基类中的具体方法
    def loaded(self):
        return bool(self.inspect())

    # 抽象基类中的具体方法
    def inspect(self):
        items = []
        while True:
            try:
                items.append(self.pick())
            except LookupError:
                break
        self.load(items)
        return tuple(sorted(items))
```

使用自定义的抽象基类：

```python
# Fake继承自抽象基类Tombola，但它没有完全覆盖所有抽象方法

class Fake(Tombola):
    def pick(self):
        return 0
```

由于`Fake`没有按照约定实现集成的抽象基类`Tombola`的所有抽象方法，因此，`Fake`类也被视为一种抽象基类，不能被实例化：

```python
# Fake被认为是一种抽象基类，因此不能被实例化

f = Fake()
```

输出：

```python
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-6-121cdeb76c24> in <module>
      1 # Fake被认为是一种抽象基类，因此不能被实例化
      2
----> 3 f = Fake()

TypeError: Can't instantiate abstract class Fake with abstract methods load
```

**声明抽象基类的最简单方式是继承`abc.ABC`或其他抽象基类**

```python
import random

# 继承抽象基类Tombola
class BingoCage(Tombola):

    def __init__(self, items):
        self._randomizer = random.SystemRandom()
        self._items = []
        self.load(items)

    # 覆盖抽象方法load
    def load(self, items):
        self._items.extend(items)
        self._randomizer.shuffle(self._items)

    # 覆盖抽象方法pick
    def pick(self):
        try:
            return self._items.pop()
        except IndexError:
            raise LookupError('pick from empty BingoCage')

    def _call__(self):
        self.pick()


# 继承抽象基类Tombola
class LotteryBlower(Tombola):

    def __init__(self, iterable):
        self._balls = list(iterable)

    # 覆盖抽象方法load
    def load(self, iterable):
        self._balls.extend(iterable)

    # 覆盖抽象方法load
    def pick(self):
        try:
            position = random.randrange(len(self._balls))
        except ValueError:
            raise LookupError('pick from empty LotteryBlower')
        return self._balls.pop(position)

    # 重载抽象基类中的具体方法
    def loaded(self):
        return bool(self._balls)

    # 重载抽象基类中的具体方法
    def inspect(self):
        return tuple(sorted(self._balls))
```

---

## 虚拟子类

即使不使用继承，也可以把一个类注册为抽象基类的虚拟子类。这样做时，我们必须保证注册的类忠实地实现了抽象基类定义的接口，而Python会选择相信我们，从而不做检查【即便是在实例化的时候也不检查】。为了避免运行出错，虚拟子类要实现所需的全部方法，否则，代码运行时会出现异常而中断程序。

**注册为虚拟子类的类不会从抽象基类中继承任何方法或属性。**

```python
# 注册虚拟子类的语法
@Tombola.register
class TomboList(list):
    # 由于TomboList注册为了抽象基类Tombola的虚拟子类，
    # 因此，它不会继承Tombola的任何方法和属性，它的__init__方法继承自list

    # 重载Tombola的抽象方法pick()
    def pick(self):
        if self: # 继承list类的__bool__方法
            position = random.randrange(len(self))
            return self.pop(position)
        else:
            raise LookupError('pop from empty TomboList')

    # 重载Tombola的抽象方法load()[使用list的extend方法代替]
    load = list.extend

    # 重载Tombola的具体方法loaded()
    def loaded(self):
        return bool(self)

    # 重载Tombola的具体方法inspect()
    def inspect(self):
        return tuple(sorted(self))
```

虚拟子类`TomboList`实现了抽象基类`Tombola`的所有方法【包括抽象方法和具体方法】。

**注册之后，可以使用`issubclass`和`isinstance`函数判断`TomboList`是不是`Tombola`的子类。**

类的继承关系在一个特殊的类属性中指定【`__mro__`】(Method Resolution Order for short)

测试：

```python
print(issubclass(TomboList, Tombola))
t = TomboList(range(100))
print(isinstance(t, Tombola))
print(TomboList.__mro__)
```

输出：

```python
True
True
(__main__.TomboList, list, object)
```

### 子类测试

**`__subclass__()`返回类的直接子类列表，不含虚拟子类。**

```python
for real_class in Tombola.__subclasses__():
    print(real_class)
```

输出：

```python
<class '__main__.Fake'>
<class '__main__.BingoCage'>
<class '__main__.LotteryBlower'>
```

### 使用register的方式

**虽然现在我们可以把`register`当做装饰器使用了，但是更常见的做法还是把它当做函数来调用，用于注册在其他地方定义的类。**

在 `collections.abc` 模块的源码中，是这样把内置类型 `tuple`、`str`、`range` 和 `memoryview` 注册为 `Sequence` 的虚拟子类的：

```python
Sequence.register(tuple)
Sequence.register(str)
Sequence.register(range)
Sequence.register(memoryview)
```

例如，将`TomboList`注册为`Tombola`的虚拟子类，通常这样做：

```python
Tombola.register(TomboList)
```

## 总结

+ 唯一推荐使用的抽象基类方法装饰器是`@abstractmethod`，其它装饰器已经废弃了；
+ 不要自己定义抽象基类，除非你要构建允许用户扩展的框架——然而大多数情况并非如此。日常使用中，我们与抽象基类的联系应该是创建现有抽象基类的子类，或者使用现有的抽象基类注册；
+ 使用抽象基类时，经常会遇到多重继承，而且是不可避免的