# 上下文管理器和else

## else

### `for`与`else`：

仅当`for`循环运行完毕（即`for`循环没有被`break`语句中断）才运行`else`语句块

### `while`与`else`：

仅当`while`循环因为条件为假而退出时（即`while`循环没有被`break`语句终止）才运行`else`语句块

在所有情况下，如果异常或者`return`，`break`，或`continue`语句导致控制权跳到了复合语句的主块之外，`else`子句也会被跳过。

### `try`与`else`：

仅当`try`块中没有异常抛出时才运行`else`块。

```python
try:
    dangerous_call()
except OSError:
    log('OSError ...')
else:
    after_call()
```

只有`try`块不抛出异常，才会执行`after_call()`。

**总而言之：`else`表达的意思是：“运行这个循环，然后做那件事”**

---

## with

`with`语句会设置一个临时的上下文，交给上下文管理其对象控制，并负责清理上下文。`with`语句的目的是简化`try/finally`模式。这种模式用于保证一段代码运行完毕后执行某项操作，即便那段代码由于异常，`return`语句或者`sys.exit()`调用而终止，也会执行指定的操作。

上下文管理器协议包含`__enter__`和`__exit__`两个方法。`with`语句开始运行时，会在上下文管理器对象上调用`__enter__`方法。`with`语句运行结束后，会在上下文管理器对象上调用`__exit__`方法，以此扮演`finally`的角色。

### 栗子

```Python
class LookingGlass:

    # 除self之外，Python调用__enter__方法时不传入其他参数
    def __enter__(self):
        import sys
        self.original_write = sys.stdout.write
        # 为sys.stdout.write打猴子补丁
        sys.stdout.write = self.reverse_write
        # 返回一个字符串，这样才有内容存入目标变量
        return 'JABBERWOCKY'

    def reverse_write(self, text):
        self.original_write(text[::-1])

    def __exit__(self, exc_type, exc_value, traceback):
        import sys
        sys.stdout.write = self.original_write
        if exc_type is ZeroDivisionError:
            print('Please DO NOT divide by zero')
            return
```

测试：

```Python
with LookingGlass() as what:
    print('Alice, Kitty and Snowdrop')
    print(what)

print(what)
```

输出：

```Python
pordwonS dna yttiK ,ecilA
YKCOWREBBAJ

JABBERWOCKY
```

不管控制流程以哪种方式退出`with`块，都会在上下文管理器对象上调用`__exit__`方法，而不是在`__enter__`方法返回的对象上调用。

**解释器调用`__enter__`方法时**，除了隐式的`self`之外，不会传入任何参数。

传给`__exit__`方法的三个参数：

1. `exc_type`:异常类（例如，`ZeroDivisionError`）
2. `exc_value`:异常实例。有时会有参数传给异常构造方法，例如错误消息，这些参数可以使用`exc_value.args`获取。
3. `traceback`:`traceback`对象。

如果一切正常，Python调用`__exit__`方法时传入的参数是`None`，`None`，`None`；如果抛出了异常，这三个参数是异常数据。

`__exit__`方法返回`True`，是为了告诉解释器异常已经处理了，此时解释器会压制异常（异常不会向上层传递），程序继续向后执行；如果`__exit__`方法返回`None`，或者`True`之外的值，`with`块中的任何异常将会向上冒泡。

---

## contextlib模块

### @contextmanager

这个装饰器把简单的生成器函数变成上下文管理器，这样就不用创建类去实现管理器协议了。

在使用`@contextmanager`装饰器的生成器中，`yield`语句的作用是把函数的定义体分成两部分：`yield`语句前面的所有代码在`with`块开始时（即解释器调用`__enter__`方法时）执行，`yield`语句后面的代码在`with`块结束时（即调用`__exit__`方法时）执行。

```python
import contextlib

@contextlib.contextmanager
def looking_glass_1():
    import sys
    original_write = sys.stdout.write

    def reverse_write(text):
        original_write(text[::-1])

    sys.stdout.write = reverse_write
    # 产出一个值，这个值会绑定到with语句中as子句的目标变量上。执行with块中的代码时，这个函数会在这一点暂停
    yield 'JABBERWOCKY'
    # 控制权一旦跳出with块，继续执行yield语句之后的代码
    sys.stdout.write = original_write
```

测试：

```python
with looking_glass_1() as what:
    print('Alice, Kitty and Snowdrop')
    print(what)

print(what)
```

输出：

```Python
pordwonS dna yttiK ,ecilA
YKCOWREBBAJ

JABBERWOCKY
```

`@contextmanger`装饰器会把函数包装成实现`__enter__`和`__exit__`方法的类。

这个类的`__enter__`方法有如下的作用：

1. 调用生成器函数，保存生成器对象（这里把它称为gen）
2. 调用`next(gen)`，执行到`yield`关键字所在的位置
3. 返回`next(gen)`产出的值，以便把产出的值绑定到`with/as`语句中的目标变量上

`with`块终止时，`__exit__`方法会做以下几件事：

1. 检查有没有把异常传给`exc_type`；如果有，调用`gen.throw(exception)`，在生成器函数定义体中包含`yield`关键字的那一行抛出异常
2. 否则，调用`next(gen)`，继续执行生成器函数定义体中`yield`语句之后的代码

`with`块终止时，如果出现异常，不会执行`yield`语句之后的代码，这可能会存在安全隐患。就本例而言，如果`with`语句发生异常，那么`sys.stdout.write`将永远无法恢复成原来的状态。

修改后的程序如下：

```Python
@contextlib.contextmanager
def looking_glass_2():
    import sys
    original_write = sys.stdout.write

    def reverse_write(text):
        original_write(text[::-1])

    sys.stdout.write = reverse_write
    msg = ''
    try:
        yield 'JABBERWOCKY'
    except ZeroDivisionError:
        msg = 'Please DO NOT division by zero!'
    # 即使发生异常，依然能做一些收尾工作
    finally:
        sys.stdout.write = original_write
        if msg: print(msg)
```

`@contextlib.contextmanager`装饰器提供的`__exit__`方法默认行为是：假定发给生成器的所有异常都得到了处理(即默认`__exit__`返回`True`），因此会对异常进行压制（进行压制的意思是不会向上冒泡异常，如果上层有对异常处理的机制，因为捕获不到也就没法进行处理）。如果真实情况并非如此，那么就会立马抛出异常从而中断之后代码的执行。若不想`@contextmanager`压制异常，必须在被装饰的函数中显式重新抛出异常，以供上层处理异常的代码处理，要么在上下文管理器中自行处理。

**重点：使用`@contextmanager`装饰器时，要把`yield`语句放在`try/finally`语句中。**

在`@contextmanager`装饰器的生成器中，`yield`与迭代没有任何关系，在这里生成器函数的作用更像协程：执行到某一点时暂停，让客户代码运行，直到客户让协程继续做事。

`with`语句块中发生异常的语句之后的代码不会执行。

测试：

```Python
with looking_glass_2() as what:
    print('ABCDEFG')
    print('='*20, ' reset sys.stdout.write '[::-1], '='*20)
    print(what)
    
print('ssssdff')
```

输出：

```Python
GFEDCBA
====================  reset sys.stdout.write  ====================
YKCOWREBBAJ
ssssdff
```

测试：

```Python
with looking_glass_2() as what:
    print('ABCDEFG')
    raise ZeroDivisionError
    # 发生异常之后的代码不会被执行
    print('='*20, ' reset sys.stdout.write '[::-1], '='*20)
    print(what)

print('ssssdff')
```

输出：

```pyhton
GFEDCBA
Please DO NOT division by zero!
ssssdff
```

上下文管理器中没有异常处理逻辑时，测试：

```Python
with looking_glass_1() as what:
    print('ABCDEFG')
    raise ZeroDivisionError
    print(what)

print('ssssdff')
```

输出：

```Python
ABCDEFG
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
<ipython-input-16-e096f6f07c5b> in <module>
      1 with looking_glass_1() as what:
      2     print('ABCDEFG')
----> 3     raise ZeroDivisionError
      4     print(what)
      5 

ZeroDivisionError: 
```

**总而言之：`@contextmanger`装饰器能把一个包含`yield`语句的简单生成器变成上下文管理器———这比定义一个至少包含两种方法的类要简洁。**