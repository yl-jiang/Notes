# 可迭代的对象、迭代器、生成器

+ **所有生成器都是迭代器，因为生成器完全实现了迭代器接口**
+ **迭代器用于从集合中取出元素，而生成器用于生成元素**

```python
class Sentence:
    RE_WORD = re.compile('\w+')
    def __init__(self, text):
        self.text = text
        self.words = self.RE_WORD.findall(text)

    def __getitem__(self, index):
        return self.words[index]

    def __len__(self):
        return len(self.words)

    def __repr__(self):
        return "Sentence(%s)" % reprlib.repr(self.text)
```

测试：

```python
s = Sentence("The time has come, the Walrus said,")
for w in s:
    print(w)

print(list(s))
```

输出：

```python
The
time
has
come
the
Walrus
said

----------------------------------------

['The', 'time', 'has', 'come', 'the', 'Walrus', 'said']
```

**序列可迭代的原因：`iter`函数**

解释器需要迭代对象`x`时，会自动调用`iter(x)`，任何Python序列都可以迭代的原因是，他们都实现了`__getitem__()`方法。

内置的`iter()`函数按照以下步骤发挥作用：

1. 检查对象是否实现了`__iter__()`方法，如果实现了，就调用它，获取一个迭代器；
2. 如果没有实现`__iter__()`方法，但是实现了`__getitem__()`方法，Python会创建一个迭代器，尝试按顺序获取元素；
3. 如果以上尝试都宣告失败，Python抛出`TypeError`异常

检查对象是否可迭代，最准确的方法是使用`iter(x)`函数，如果不可迭代，再处理`TypeError`异常。最好的方式是调用`isinstance(x, abc.Iterator)`。

---

## 可迭代对象

使用 `iter` 内置函数可以获取迭代器的对象。如果对象实现了能返回迭代器的 `__iter__` 方法，那么对象本身就是可迭代的了。实现了 `__getitem__` 方法，而且其参数是从零开始的索引，这种对象也可以迭代。

可迭代的对象和迭代器之间的关系：Python 从可迭代的对象中获取迭代器。

```python
s = 'ABC' 
for char in s:  
    print(char) 
```

输出：

```python
A
B
C
```

字符串`'ABC'`是可迭代对象。

---

## 将迭代器对象和迭代器分开

```python
class Sentence:
    RE_WORD = re.compile("\w+")

    def __init__(self, text):
        self.text = text
        self.wprds = RE_WORD.findall(text)

    def __repr__(self):
        return "Sentence(%s)" % reprlib.repr(se;f.text)

    # 只实现了__iter__()方法，因此Sentence是可迭代对象
    def __iter__(self):
        return SentenceIterator(self.words)


class SentenceIterator:
    """
    SentenceIterator实现了__next__()和__iter__()方法，因此SentenceIterator是迭代器
    """

    def __init__(self, words):
        self.words = words
        self.index = 0

    def __next__(self):
        try:
            word = self.word[self.index]
        except IndexError:
            raise StopIteration()
        self.index += 1

    # 迭代器的__iter__()方法应该返回self，这样当外部每次使用iter(xxx)时，可以返回一个新的迭代器
    def __iter__(self):
        return self
```

迭代器应该(也只需)实现 `__next__` 和 `__iter__` 两个方法，这么做能让迭代器通过`issubclass(yourIterator, abc.Iterator)`测试。

实现一个**可迭代对象**，需要实现`__iter__()`方法，该方法返回一个新的迭代器。

而实现一个**迭代器**，要实现`__iter__()`方法和`__next__()`方法，其中`__next__()`方法返回单个元素，`__iter__()`方法返回迭代器本身。

**迭代器可以迭代，但是可迭代对象不是迭代器。可迭代的对象一定不能同时是自身的迭代器，也就是说，可迭代的对象必须实现`__iter__()`方法，但是最好不要实现`__next__()`方法。另一方面，迭代器应该一直可以迭代，迭代器的`__iter__()`方法应该返回本身。**

---

## 使用生成器函数代替SentenceIterator类(`yield`)

Python中始终可以使用生成器这个语言结构履行迭代器的基本职责：遍历集合，并从中产出元素。

因此，实现一个可迭代对象，有以下两种方式：

1. 实现`__iter__()`方法，且该方法返回一个迭代器对象(需另外单独实现)
2. 实现`__iter__()`方法，且该方法返回一个生成器(使用生成器函数或者生成器表达式)

```python
class Sentence:

    RE_WORD = re.compile('\w+')

    def __init__(self, text):
        self.text = text
        self.words = RE_WORD.findall(text)

    def __repr__(self):
        return 'Sentence(%s)' % reprlib.repr(text)

    # Sentence只实现了__iter__()方法，因此Sentence是一个可迭代对象，但是，由于使用了生成器函数，因此它具有了迭代器的功能
    def __iter__(self):
        for word in self.words:
            yield word
        # 不管有没有 return 语句，生成器函数都不会抛出 StopIteration 异常，而是在生成完所有值之后会直接退出
        return
```

只要 Python 函数的定义体中有 `yield` 关键字，该函数就是生成器函数。调用生成器函数时，会返回一个生成器对象。也就是说，生成器函数是生成器工厂。

生成器函数会创建一个生成器对象，包装生成器函数的定义体。把生成器传给 `next(...)` 函数时，生成器函数会向前，执行函数定义体中的下一个 `yield` 语句，返回产出的值，并在函数定义体的当前位置暂停。

### yield from

如果生成器函数需要产出另一个生成器生成的值，传统的解决方法是使用`for`循环：

```python
def chain(*iterables):
    for it in iterables:
        for item in it:
            yield item
```

现在可以使用`yield from`：

```python
def chain(*iterables):
    for it in iterables:
        yield from it
```

`yield from`完全替换了内层的`for`循环。当把生成器当做协程使用时，这个通道特别重要。

---

## 惰性实现

```python
class Sentence:
    RE_WORD = re.compile('\w+')

    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return "Sentence(%s)" % reprlib.repr(self.text)

    # 将__iter__()定义为一个生成器函数
    def __iter__(self):
        # 使用了re模块的惰性匹配，不一次性构建self.words列表
        for match in RE_WORD.finditer(self.text):
            yield match.group()
```

惰性实行是尽可能的延后生成值。

---

## 用生成器表达式代替yield

**重点：生成器是迭代器。**

```python
class Sentence:
    RE_WORD = re.compile('\w+')

    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return "Sentence(%s)" % reprlib.repr(self.text)

    # 使用生成器表达式代替生成器函数
    def __iter__(self):
        return (match.group() for match in RE_WORD.finditer(self.text))
```

生成器表达式是语法糖，可以替换简单的生成器函数。

---

## 使用生成器函数实现特殊的`__iter__()`方法

```python
class ArithmeticProgression:

    def __init__(self, begin, step, end=None):
        self.begin = begin
        self.step = step
        self.end = end

    def __iter__(self):
    """
    按照一定的步长产出下一个值
    """
        # 'type(self.begin + self.step)'是为了得到self.begin或者self.end其中某一个的数据类型，作为返回结果的类型
        result = type(self.begin + self.step)(self.begin)
        forever = self.end is None
        self.index = 0
        while forever or result < self.end:
            yield result
            self.index += 1
            result = self.begin + self.index * self.step
```

测试：

```python
ap = ArithmeticProgression(0, 1, 3)
print（list(ap)
```

输出：

```python
[0, 1, 2]
```

使用生成器函数，可以在产生元素时，加上限制条件。

---

## 多利用内置的itertools模块

`itertools.takewhile` 会生成一个生成器并返回，返回的生成器在指定的条件计算结果为`False`时停止。

```python
import itertools

def aritprog_gen(begin, step, end=None):
    first = type(begin + step)(begin)
    ap_gen = itertools.count(first, step)
    if end is not None:
        ap_gen = itertools.takewhile(lambda n: n < end, ap_gen)
    return ap_gen
```

好用的itertools函数：

+ `itertools.chain()`
+ `itertools.product()`
+ `itertools.zip_longest()`
+ `itertools.zip()`
+ `itertools.repeat()`

---

## 内置iter()函数

`iter` 函数还有一个鲜为人知的用法：传入两个参数，使用常规的函数或任何可调用的对象创建迭代器。这样使用时，第一个参数必须是可调用的对象，用于不断调用（没有参数），产出各个值；第二个值是哨符，这是个标记值，当可调用的对象返回这个值时，触发迭代器抛出 `StopIteration` 异常，而不产出哨符。

```python
import random
def sample():
    return random.randint(1, 6)

# 掷拥有6个面的色子，直到得到数字1为止
sampler_1 = iter(sample, 1)

for num in sampler_1:
    print(num)
```

输出：

```python
2
6
3
2
5
6
```

有个实用的例子。这段代码逐行读取文件，直到遇到空行或者到达文件末尾为止：

```python
with open('data.txt') as fp:
    for line in iter(fp.readline, '\n'):
        process_func(line)
```