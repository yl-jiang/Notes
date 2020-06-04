# 继承

使用C语言编写的内置类型方法不会调用子类覆盖的方法。例如，`dict`的子类覆盖的`__getitem__()`方法不会被内置类型的`get()`方法调用。【这是基于性能的考量】

直接子类化内置类型（`dict`, `list`, `str`等）容易出错，因为内置类型的方法通常会忽略用户覆盖的方法。不要子类化内置类型，用户自己定义的类应该继承`collections`模块中相应的类，如，`UserDict`, `UserList`, `UserString`,这些类做了特殊设计，因此易于扩展。

---

## self

```python
class A:

    @classmethod
    def func1(cls):
        print('classmethod')

    def func2(self):
        print('instance method')
```

测试：

```python
[]In: A.func1()
classmethod

[]In: A.func2()
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-6-24ea7cd4ba02> in <module>
----> 1 A.func2()

TypeError: func2() missing 1 required positional argument: 'self'

[]In: A.func2(A)
instance method

[]In:a = A()
[]In:a.func2()
instance method
```

`class`中的`self`参数代表了当前的实例。

---

## 多重继承

```python
class A:

    def __init__(self):
        self.a = 'A'

    def ping(self):
        print("class A's ping")
        print(f"self.a = {self.a}")

class B(A):
    def __init__(self):
        self.a = 'B'

    def pong(self):
        print("class B's pong")
        print(f"self.a = {self.a}")

class C(A):

    def __init__(self):
        self.a = 'C'

    def pong(self):
        print("class C's pong")
        print(f"self.a = {self.a}")


class D(B, C):

    def __init__(self):
        self.a = 'D'

    def ping(self):
        """
        在这里调用D父类的ping()方法，解释器会首先在B中寻找ping()方法，如果B中没有，
        进而从C中寻找ping()方法，如果C中也没有，
        紧接着向B的父类【也就是A】寻找ping()方法，找到了。
        """
        super().ping()  # 明确指定使用class D父类的ping方法
        print("class D's ping")
        print(f"self.a = {self.a}")

    def pingpong(self):
        self.ping()
        super().ping()  # 根据方法解析顺序找到ping()方法。等价于 'A.ping(self)'
        self.pong()  # 根据方法解析顺序找到pong()方法。等价于 'B.pong(self)'
        super().pong()  # 根据方法解析顺序找到pong()方法。等价于 'B.pong(self)'
        """
        直接调用C的pong()方法，注意传入的是self代表D的实例
        """
        C.pong(self)  # 忽略方法解析顺序，直接使用C中的pong()方法
```

测试：

```python
[]In: d = D()
[]In: d.pong()
class B's pong
self.a = D

[]In: d.ping()
class A's ping
self.a = D
class D's ping
self.a = D

[]In: d.pingpong()
class A's ping
self.a = D
class D's ping
self.a = D
class A's ping
self.a = D
class B's pong
self.a = D
class B's pong
self.a = D
class C's pong
self.a = D
```

**多重继承中方法的查找，使用类似于宽度优先的策略。使用`super()`调用方法时，会遵循方法解析顺序。查看类的方法解析顺序，使用`classname.__mro__`。**

在子类中想把方法委托给超类，推荐的方式是使用内置的`super()`函数，这样最安全也最不容易过时。

```python
[]In: D.__mro__
(__main__.D, __main__.B, __main__.C, __main__.A, object)
```

## 总结

+ 如果类的作用仅仅是定义接口，应该明确把它定义为抽象基类
+ 如果一个类的作用是为多个不相关的子类提供方法实现，从而实现重用，但不体现“是什么”关系，应该把那个类明确定义为混入类（mixin class）。**从概念上讲，混入类不定义新类型，只是打包方法，便于重用**。混入类绝对不能实例化，而且具体类不能只继承混入类。
+ Python中没有把类声明为混入的正规方式，所以强烈推荐在名称中加入`...Mixin`后缀
