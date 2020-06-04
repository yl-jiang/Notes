# 协程

**从句法上看，协程与生成器类似，都是定义体中包含`yield`关键字的函数。从根本上把`yield`是做流程控制的方式，可以更好地理解协程。**

**协程是指一个过程，这个过程与调用方协作，产出由调用方提供的值。**

```python
def simple_coroutine():
    print('-> coroutine started')
    x = yield
    print('-> coroutine received: ', x)
```

测试：

```Python
my_coro = simple_coroutine()
print(my_coro)
print(next(my_coro))
```

输出：

```Python
<generator object simple_coroutine at 0x7f040017d7d0>
-> coroutine started
None
```

生成器调用方可以使用`.send(...)`方法发送数据，发送的数据会成为生成器函数中`yield`表达式左边的值。

继续执行上述协程：

```python
my_coro.send(42)
```

输出：

```Python
-> coroutine received:  42
---------------------------------------------------------------------------
StopIteration                             Traceback (most recent call last)
<ipython-input-3-7c96f97a77cb> in <module>
----> 1 my_coro.send(42)

StopIteration: 
```

协程可以身处四种状态中的一种，当前状态可以使用`inspect.getgeneratorstate(...)`函数确定，该函数返回下述字符串中的一个：

1. `GEN_CREATED`:等待开始执行
2. `GEN_RUNNING`:解释器正在执行
3. `GEN_SUSPENDED`:在`yield`表达式处暂停
4. `GEN_CLOSED`:执行结束

因为`send`方法的参数会成为暂停`yield`表达式的值，所以，仅当协程处于暂停状态时才能调用`send`方法。

查看协程处在那种状态：

```Python
from inspect import getgeneratorstate
# 创建协程对象后，立即把None之外的值传递给他，会出现错误
my_coro = simple_coroutine()
print(getgeneratorstate(my_coro))
my_coro.send(10)
```

输出：

```Python
GEN_CREATED
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-4-7012362bd954> in <module>
      3 my_coro = simple_coroutine()
      4 print(getgeneratorstate(my_coro))
----> 5 my_coro.send(10)

TypeError: can't send non-None value to a just-started generator
```

---

## 预激协程

最先调用`next(my_coro)`这一步通常称为“预激（prime）”协程（即，让协程向前执行到第一个`yield`表达式，准备好作为活跃的协程使用）。

```Python
def simple_coro_2(a):
    print(f'-> started: a = {a}')
    b = yield a
    print(f'-> received: b = {b}')
    c = yield a + b
    print(f'-> received: c = {c}')
```

测试：

```Python
my_coro_2 = simple_coro_2(10)
print(getgeneratorstate(my_coro_2))

# 执行到第一个yield处：
# 1.执行第一个print语句;
# 2.执行yield a(即，返回a的值)；
# 3.程序停在"b="处，等待用户输入
print(next(my_coro_2)) 
```

输出：

```Python
GEN_CREATED
-> started: a = 10
10
```

再次查看协程的状态：

```Python
print(getgeneratorstate(my_coro_2))
```

输出：

```Python
GEN_SUSPENDED
```

再次向协程发送数据，并查看协程状态：

```Python
# 1.参数b接受send发送过来的值，b = 20;
# 2.执行第二个print语句；
# 3.执行“yield a + b”（即，返回a+b的值）;
# 4.程序停在"c="处，等待用户输入
print(my_coro_2.send(20))

print(getgeneratorstate(my_coro_2))
```

输出：

```Python
-> received: b = 20
30
GEN_SUSPENDED
```

继续发送数据测试：

```Python
# 1. 参数c接收send发送过来的值， c = 30；
# 2. 执行第三个print语句；
# 3. 寻找下一个yield语句，没有找到，抛出"StopIteration"异常
print(my_coro_2.send(30))
```

输出：

```Python
-> received: c = 30
---------------------------------------------------------------------------
StopIteration                             Traceback (most recent call last)
<ipython-input-10-aee716836424> in <module>
      2 # 2. 执行第三个print语句；
      3 # 3. 寻找下一个yield语句，没有找到，抛出"StopIteration"异常
----> 4 print(my_coro_2.send(30))

StopIteration: 
```

---

## 运用协程的一个栗子

使用协程实现计算平均值：

```Python
def averager():
    total = 0.0
    count = 0
    average = None
    while True:
        term = yield average
        total += term
        count += 1
        average = total / count
```

协程的好处是，`total`和`count`声明为局部变量即可，无需使用实例属性或闭包在多次调用之间保持上下文。

测试：

```Python
coro_avg = averager()

# 1. 预激协程'averager()'
# 2. 返回average的初始值None
print(next(coro_avg))

print(coro_avg.send(10))
print(coro_avg.send(20))
print(coro_avg.send(5))
```

输出：

```Python
None

10.0
15.0
11.666666666666666
```

---

## 预激协程的装饰器

预激协程的装饰器函数:

```Python
from functools import wraps

def coroutine(func):
    # functools.wraps装饰器，保持被装饰函数在经过装饰后所有原始信息不变
    @wraps(func)  
    def primer(*args, **kwargs):
        gen = func(*args, **kwargs)
        next(gen)
        return gen
    return primer
```

将预激装饰器作用于协程：

```Python
@coroutine
def averager():
    total = 0.0
    count = 0
    average = None
    while True:
        term = yield average
        total += term
        count += 1
        average = total / count
```

测试：

```Python
# 经过预激装饰器修饰后，初始化后的装饰就已经处于暂停状态了。
coro_avg = averager()
print(getgeneratorstate(coro_avg))

print(coro_avg.send(10))
print(coro_avg.send(20))
```

输出：

```Python
GEN_SUSPENDED
10.0
15.0
```

---

## 终止协程和异常处理

正常使用协程时：

```Python
coro_avg = averager()
print(coro_avg.send(40))
print(coro_avg.send(50))
```

输出：

```Python
40.0
45.0
```

由于协程内没有进行异常处理，传入不符合要求的数据时，携程会终止。

```Python
print(coro_avg.send('spam'))
```

输出：

```Python
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-27-342721471c33> in <module>
----> 1 print(coro_avg.send('spam'))

<ipython-input-18-e0aef2af84a4> in averager()
      6     while True:
      7         term = yield average
----> 8         total += term
      9         count += 1
     10         average = total / count

TypeError: unsupported operand type(s) for +=: 'float' and 'str'
```

如果试图重新激活已经出现异常的协程，会抛出`StopIteration`。

```Python
print(coro_avg.send(60))
```

输出：

```Python
---------------------------------------------------------------------------
StopIteration                             Traceback (most recent call last)
<ipython-input-28-921f0c5ecf39> in <module>
----> 1 print(coro_avg.send(60))

StopIteration: 
```

修改协程，加入异常处理代码：

```Python
class DemoException(Exception):
    pass

# 定义一个只处理DemoException异常的协程
def demo_exc_handling():
    print('-> coroutine started')
    while True:
        try:
            x = yield
        # 只处理DemoException这一种类型的异常
        except DemoException:
            print('*** DemoException handled. Continuing ...')
        else:
            print('-> coroutine received: {!r}'.format(x))

    # 这条语句应该永远不会被执行到，因为：
    # 1. 如果发生DemoException异常，那么有专门的处理该异常的代码逻辑，处理完成后协程继续工作
    # 2. 如果发生了除DemoException之外的异常，协程会立即终止
    raise RuntimeError('This line should never run')
```

测试：

```Python
exc_coro = demo_exc_handling()
# 预激协程
next(exc_coro)
print(exc_coro.send(10))
```

输出：

```Python
-> coroutine started
-> coroutine received: 10
None
```

第一句输出是函数中`else`语句的`print`语句的输出，输出的`None`是`yield`的返回值。

---

## 关闭协程

```Python
# 使用close方法关闭协程
print(exc_coro.close())

getgeneratorstate(exc_coro)
```

输出：

```Python
None
'GEN_CLOSED'
```

---

## 协程异常处理

协程调用`.throw(SomeException)`方法，可以将指定的异常抛给协程；协程调用`.close()`方法，可以将该协程关闭。

### 处理DemoException异常

```Python
exc_coro = demo_exc_handling()
# 预激协程
next(exc_coro)  # out[]: -> coroutine started
```

正常使用协程：

```Python
print(exc_coro.send(10))
```

输出：

```Python
-> coroutine received: 10
None
```

传入`DemoException`异常给协程;

```python 
# 抛出DemoException异常给协程
print(exc_coro.throw(DemoException))
```

输出：

```Python
*** DemoException handled. Continuing ...
None
```

遇到异常并处理后，协程继续执行。

```Python
print(getgeneratorstate(exc_coro))  # out[]: 'GEN_SUSPENDED'
```

### 处理非`DemoException`异常

```python 
exc_coro = demo_exc_handling()
# 预激协程
next(exc_coro)  # out[]: -> coroutine started
```

传入非`DemoException`异常到协程：

```python 
# 抛出非DemoException异常给协程
print(exc_coro.throw(ZeroDivisionError))
```

输出：

```Python
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
<ipython-input-56-65172b3876cd> in <module>
      1 # 抛出非DemoException异常给协程
----> 2 print(exc_coro.throw(ZeroDivisionError))

<ipython-input-29-386183283adb> in demo_exc_handling()
      6     while True:
      7         try:
----> 8             x = yield
      9         # 只处理DemoException这一种类型的异常
     10         except DemoException:

ZeroDivisionError: 
```

查看此时协程的状态：

```Python
print(getgeneratorstate(exc_coro))  # out[]: 'GEN_CLOSED'
```

由于`exc_coro`没有处理非`DemoException`异常的能力，因此协程会终止。

如果不管协程如何结束都想做一些清理工作，要把协程定义体中相关的代码放入`try/finally`块中。

```python 
def demo_finally():
    print('-> coroutine started')
    try:
        while True:
            try:
                x = yield
            # 只处理DemoException这一种类型的异常
            except DemoException:
                print('*** DemoException handled. Continuing ...')
            else:
                print('-> coroutine received: {!r}'.format(x))
    finally:
        # 不用调用corotine.close()方法，因为遇到它不能处理的异常会自行关闭
        print('-> coroutine ending.')
```

---

## 协程的返回值

如果协程中`yield`表达式的右边没有任何内容时，默认每次激活协程会返回一个`None`。

```python
from collections import namedtuple

Result = namedtuple('Result', ['count', 'average'])

def averager():
    total = 0.0
    count = 0
    while True:
        term = yield
        # 当协程接收到用户输入的None，代表计算结束
        if term is None:
            break
        total += term
        count += 1
    return Result(count, total / count)
```

测试：

```Python
coro_avg = averager()
next(coro_avg)
print(coro_avg.send(10))
print(coro_avg.send(30))
print(coro_avg.send(40))
print(coro_avg.send(None))
```

输出：

```Python
None
None
None

---------------------------------------------------------------------------
StopIteration                             Traceback (most recent call last)
<ipython-input-64-b0d46019c513> in <module>
----> 1 print(coro_avg.send(None))

StopIteration: Result(count=3, average=26.666666666666668)
```

传给协程`None`时：

1. 协程结束
2. 返回结果
3. 生成器对象抛出`StopIteration`异常，异常对象的`value`属性保存着返回值

### 获取协程的返回值

获取协程的返回值要绕个圈子。

```Python
from collections import namedtuple

Result = namedtuple('Result', ['count', 'average'])

def averager():
    total = 0.0
    count = 0
    while True:
        term = yield
        # 当协程接收到用户输入的None，代表计算结束
        if term is None:
            break
        total += term
        count += 1
    return Result(count, total / count)


if __name__ == '__main__':
    coro_avg = averager()
    next(coro_avg)
    coro_avg.send(10)
    coro_avg.send(20)
    coro_avg.send(30)

    try:
        coro_avg.send(None)
    except StopIteration as exc:
        result = exc.value

    print(result)
```

输出：

```Python
Result(count=3, average=20.0)
```

---

## yield from

在生成器`gen`中使用`yield from subgen()`:`subgen()`会获得控制权，把产出的值传给`gen`的调用方，即调用方可以直接控制`subgen`。与此同时，`gen`会阻塞，等待`subgen`终止。

```Python
def gen():
    for c in 'AB':
        yield c
    for i in range(1, 3):
        yield i
```

测试：

```Python
print(list(gen()))
```

输出：

```Python
['A', 'B', 1, 2]
```

使用`yield from`修改上述程序：

```Python
def gen():
    yield from 'AB'
    yield from range(1, 3)
```

测试：

```Python
print(list(gen()))
```

输出：

```Python
['A', 'B', 1, 2]
```

`yield from x`表达式对`x`做的第一件事就是调用`iter(x)`，从中获得迭代器，`x`可以是任意可迭代的对象。

`yield from`的主要功能是打开双向通道，把最外层的调用方与最内层的子生成器连接起来，这样二者就可以直接发送和产出值，还可以直接传入异常，而不用在位于中间的协程中添加大量处理异常的样板代码。

引入`yield from`结构的目的是为了支持实现了`__next__`、`send`、 `close`和`throw`方法的生成器（也就是说为了更方便的实现协程）。

```Python
from collections import namedtuple

Result = namedtuple('Result', ['count', 'average'])

def averager():
    total = 0.0
    count = 0
    while True:
        term = yield
        # 当协程接收到用户输入的None，代表计算结束
        if term is None:
            break
        total += term
        count += 1
    return Result(count, total / count)

def grouper(results, key):
    while True:
        results[key] = yield from averager()

def main(data):
    results = {}
    for key, values in data.items():
        group = grouper(results, key)
        next(group)
        for value in values:
            group.send(value)
        group.send(None)
    return results
```

`main()`函数的执行逻辑:

1. 运行到`main()`函数的`group = grouper(results, key)`时，程序被协程`averager()`接管，此时`group`代表了协程`averager()`；
2. 当内层`for`循环（`'for value in values:'`）结束（即取完values中的所有值）后，`group`实例依旧在`yield from`表达式处暂停，因此，`grouper`函数定义体中的赋值语句`result[key]`还没有执行；
3. 紧接着输入`None`，终止`averager`实例，抛出`'StopIteration'`异常并向上冒泡，控制权回到函数`grouper()`。
4. 之后，`yield from`表达式的值是协程终止时传给`'StopIteration'`异常的第一个参数（`StopIteration.value`），并将该值绑定到`results[key]`。`grouper()`接收到`''StopIteration''`异常，而该函数内没有异常处理代码，所以继续向上冒泡异常；
5. `'StopIteration'`异常并冒泡到外层`for`循环（`'for key, values in data.items():'`），该层`for`循环接受到`'StopIteration'`异常后，认为迭代完成，压制异常，开始下一次循环
6. 外层`for`循环重新迭代时，会新建一个`grouper`实例，然后绑定到`group`变量上，前一个`grouper`实例被垃圾回收程序回收

测试：

```Python
import random
data = {'girls;kg':[40.9, 38.5, 44.3, 42.2, 45.2, 41.7, 44.5, 38.0, 40.6, 44.5],
        'girls;m':[1.6, 1.51, 1.4, 1.3, 1.41, 1.39, 1.33, 1.46, 1.45, 1.43],
        'boys;kg':[39.0, 40.8, 43.2, 40.8, 43.1, 38.6, 41.4, 40.6, 36.3],
        'boys;m':[1.38, 1.5, 1.32, 1.25, 1.37, 1.48, 1.25, 1.49, 1.46]} 

results = main(data)
print(results)
```

输出：

```Python
{'girls;kg': Result(count=10, average=42.040000000000006),
 'girls;m': Result(count=10, average=1.4279999999999997),
 'boys;kg': Result(count=9, average=40.422222222222224),
 'boys;m': Result(count=9, average=1.3888888888888888)}
```

---

## 使用协程做离散时间仿真

```python
import collections
Event = collections.namedtuple('Event', ['time', 'proc', 'action'])

def taxi_process(ident, trips, start_time=0):
    time = yield Event(start_time, ident, 'leave garage')
    for i in range(trips):
        time = yield Event(time, ident, 'pick up passenger')
        time = yield Event(time, ident, 'drop off passenger')
    yield Event(time, ident, 'going home')


DEPARTURE_INTERVAL = 5
SEARCH_DURATION = 5
TRIP_DURATION = 20
DEFAULT_END_TIME = 180
DEFAULT_NUMBER_OF_TAXIS = 3


import random

def compute_duration(previous_action):
    if previous_action in ['leave garage', 'drop off passenger']:
        interval = SEARCH_DURATION
    elif previous_action == 'pick up passenger':
        interval = TRIP_DURATION
    elif previous_action == 'going home':
        interval = 1
    else:
        raise ValueError(f'Unknown previous_action: {previous_action}')
    return int(random.expovariate(1 / interval)) + 1


import queue

class Simulator:
    
    def __init__(self, procs_map):
        # 优先队列是离散事件仿真的基础构件，可按各个事件排定的时间顺序取出
        self.events = queue.PriorityQueue()
        # 创建一个dict副本，防止修改用户传进来的数据
        self.procs = dict(procs_map)
        
    def run(self, end_time):
        # 预激各个taxi协程，并将它们放入到主循环
        for _, proc in sorted(self.procs.items()):
            first_event = next(proc)
            self.events.put(first_event)
        
        # 初始化主循环
        sim_time = 0
        # 结束主循环的条件：
        # 1. 一天结束
        # 2. 各个出租车都提前完成一天的任务量
        while sim_time < end_time:
            if self.events.empty():
                print('*** end of events ***')
                break
                
            # 取出当前需要处理的协程（时间值最小的那个）
            current_event = self.events.get()
            sim_time, proc_id, previous_action = current_event
            # 每个taxi协程以不同的缩进打印，方便查阅
            print('taxi: ', proc_id, proc_id * '  ', current_event)
            # 获取当前协程
            activate_proc = self.procs[proc_id]
            # 更新主循环
            next_time = sim_time + compute_duration(previous_action)
            try:
                # 给当前协程互动，传送相应的值
                next_event = activate_proc.send(next_time)
            # 如果当前协程已经完成了所有任务，则以后不再处理该协程
            except StopIteration:
                del self.procs[proc_id]
            else:
                # 如果该协程没有终结，且用户正确的传入了数据，那么将它放到主循环的有限队列中
                self.events.put(next_event)
        # 主循环的结束是因为一天结束(这种情况可能有些出租车并没有完成所有任务量)，才会执行代码块
        else:
            msg = f'*** end of simulation time: {self.events.qsize()} events pending'
            print(msg)
```

测试：

```python
def main(end_time = DEFAULT_END_TIME, 
         num_taxis = DEFAULT_NUMBER_OF_TAXIS, 
         seed = 3):
    if seed is not None:
        random.seed(seed)
    # 创建了三辆出租车一天的行程(协程)，他们分别相隔5分钟开始一天的工作
    taxis = {i: taxi_process(i, (i + 1) * 2, i * DEPARTURE_INTERVAL)
             for i in range(num_taxis)}
    sim = Simulator(taxis)
    sim.run(end_time)


if __name__ == '__main__':
    main()
```

输出：

```python
taxi:  0  Event(time=0, proc=0, action='leave garage')
taxi:  0  Event(time=2, proc=0, action='pick up passenger')
taxi:  1    Event(time=5, proc=1, action='leave garage')
taxi:  1    Event(time=8, proc=1, action='pick up passenger')
taxi:  2      Event(time=10, proc=2, action='leave garage')
taxi:  2      Event(time=15, proc=2, action='pick up passenger')
taxi:  2      Event(time=17, proc=2, action='drop off passenger')
taxi:  0  Event(time=18, proc=0, action='drop off passenger')
taxi:  2      Event(time=18, proc=2, action='pick up passenger')
taxi:  2      Event(time=25, proc=2, action='drop off passenger')
taxi:  1    Event(time=27, proc=1, action='drop off passenger')
taxi:  2      Event(time=27, proc=2, action='pick up passenger')
taxi:  0  Event(time=28, proc=0, action='pick up passenger')
taxi:  2      Event(time=40, proc=2, action='drop off passenger')
taxi:  2      Event(time=44, proc=2, action='pick up passenger')
taxi:  1    Event(time=55, proc=1, action='pick up passenger')
taxi:  1    Event(time=59, proc=1, action='drop off passenger')
taxi:  0  Event(time=65, proc=0, action='drop off passenger')
taxi:  1    Event(time=65, proc=1, action='pick up passenger')
taxi:  2      Event(time=65, proc=2, action='drop off passenger')
taxi:  2      Event(time=72, proc=2, action='pick up passenger')
taxi:  0  Event(time=76, proc=0, action='going home')
taxi:  1    Event(time=80, proc=1, action='drop off passenger')
taxi:  1    Event(time=88, proc=1, action='pick up passenger')
taxi:  2      Event(time=95, proc=2, action='drop off passenger')
taxi:  2      Event(time=97, proc=2, action='pick up passenger')
taxi:  2      Event(time=98, proc=2, action='drop off passenger')
taxi:  1    Event(time=106, proc=1, action='drop off passenger')
taxi:  2      Event(time=109, proc=2, action='going home')
taxi:  1    Event(time=110, proc=1, action='going home')
*** end of events ***
```