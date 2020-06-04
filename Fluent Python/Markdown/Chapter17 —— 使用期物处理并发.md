# 使用期物处理并发

本章主要讨论 Python 3.2 引入的 `concurrent.futures` 模块。

期物（future）的概念：future指一种对象，表示异步执行的操作。

在I/O密集型应用中，并发策略（使用线程或asyncio包），吞吐量都比依次执行的代码高很多。

## 进程和线程的区别

作者：zhonyong

链接：https://www.zhihu.com/question/25532384/answer/81152571

来源：知乎

首先来一句概括的总论：**进程和线程都是一个时间段的描述，是CPU工作时间段的描述。**

**下面细说背景**：CPU+RAM+各种资源（比如显卡，光驱，键盘，GPS, 等等外设）构成我们的电脑，但是电脑的运行，实际就是CPU和相关寄存器以及RAM之间的事情。

**一个最最基础的事实**：CPU太快，太快，太快了，寄存器仅仅能够追的上他的脚步，RAM和别的挂在各总线上的设备完全是望其项背。那当多个任务要执行的时候怎么办呢？轮流着来?或者谁优先级高谁来？**不管怎么样的策略，一句话就是在CPU看来就是轮流着来。**

**一个必须知道的事实**：执行一段程序代码，实现一个功能的过程介绍 ，当得到CPU的时候，相关的资源必须也已经就位，就是显卡啊，GPS啊什么的必须就位，然后CPU开始执行。这里除了CPU以外所有的就构成了这个程序的执行环境，也就是我们所定义的程序上下文。当这个程序执行完了，或者分配给他的CPU执行时间用完了，那它就要被切换出去，等待下一次CPU的临幸。在被切换出去的最后一步工作就是保存程序上下文，因为这个是下次他被CPU临幸的运行环境，必须保存。

**串联起来的事实**：前面讲过在CPU看来所有的任务都是一个一个的轮流执行的，具体的轮流方法就是：先加载程序A的上下文，然后开始执行A，保存程序A的上下文，调入下一个要执行的程序B的程序上下文，然后开始执行B,保存程序B的上下文。。。。

========= 重要的东西出现了========

进程和线程就是这样的背景出来的，两个名词不过是对应的CPU时间段的描述，名词就是这样的功能。

**进程就是包换上下文切换的程序执行时间总和 = CPU加载上下文+CPU执行+CPU保存上下文。**

线程是什么呢？

进程的颗粒度太大，每次都要有上下的调入，保存，调出。如果我们把进程比喻为一个运行在电脑上的软件，那么一个软件的执行不可能是一条逻辑执行的，必定有多个分支和多个程序段，就好比要实现程序A，实际分成 a，b，c等多个块组合而成。那么这里具体的执行就可能变成：程序A得到CPU =》CPU加载上下文，开始执行程序A的a小段，然后执行A的b小段，然后再执行A的c小段，最后CPU保存A的上下文。这里a，b，c的执行是共享了A的上下文，CPU在执行的时候没有进行上下文切换的。这里的a，b，c就是线程，也就是说线程是共享了进程的上下文环境，的更为细小的CPU时间段。

到此全文结束，再一个总结：进程和线程都是一个时间段的描述，是CPU工作时间段的描述，不过是颗粒大小不同。

---

## 网络下载的三种风格

### 顺序下载

```python
import os
import time
import sys
import requests

POP20_CC = ('CN IN US ID BR PK NG BD RU JP MX PH VN ET EG DE IR TR CD FR').split()
print(f'len(POP20_CC) = {len(POP20_CC)}')

BASE_URL = 'http://flupy.org/data/flags'
DEST_DIR = './downloads'


def save_flag(img, filename):
    path = os.path.join(DEST_DIR, filename)
    with open(path, 'wb') as fp:
        fp.write(img)

def get_flag(cc):
    url = f'{BASE_URL}/{cc.lower()}/{cc.lower()}.gif'
    resp = requests.get(url)
    return resp.content

def show(text):
    print(text, end=' ')
    sys.stdout.flush()

def download_many(cc_list):
    for cc in sorted(cc_list):
        image = get_flag(cc)
        show(cc)
        save_flag(image, cc.lower() + '.gif')

def main(download_many):
    t0 = time.time()
    count = download_many(POP20_CC)
    elapsed = time.time() - t0
    print(f'\n{count} flags downloaded in {elapsed:.2f}s')
```

测试：

```python
>>> main(download_many)
BD BR CD CN DE EG ET FR ID IN IR JP MX NG PH PK RU TR US VN
None flags downloaded in 43.50s
```

## 使用concurrent.futures模块下载

`concurrent.futures`模块的主要特色是`ThreadPoolExecutor`和`ProcessPoolExecutor`类，这两个类实现的接口能分别在不同的线程或进程中执行可调用对象。

```python
MAX_WORKERS = 20
from concurrent import futures

def download_one(cc):
    image = get_flag(cc)
    show(cc)
    save_flag(image, cc.lower() + '.gif')
    return cc

def download_many(cc_list):
    # 设定 ThreadPoolExecutor 类最多使用几个线程。
    workers = min(MAX_WORKERS, len(cc_list))
    # 使用工作的线程数实例化 ThreadPoolExecutor 类
    with futures.ThreadPoolExecutor(workers) as executor:
        # Executor.map方法返回值是一个迭代器，迭代器的__next__方法调用各个期物result方法，因此我们最后得到的是各个期物的结果，而非期物本身
        res = executor.map(download_one, sorted(cc_list))
    return len(list(res))
```

本例中的`download_one`函数其实是上例中`download_many` 函数的 `for` 循环体。编写并发代码时经常这样重构： **把依序执行的 `for` 循环体改成函数，以便并发调用。**

`concurrent.futures` 模块的主要特色是 `ThreadPoolExecutor` 和 `ProcessPoolExecutor` 类，这两个类实现的接口能分别在不同的线程 或进程中执行可调用的对象。这两个类在内部维护着一个工作线程或进 程池，以及要执行的任务队列。

`ThreadPoolExecutor.map()`函数的作用与内置的`map`函数类似，不过`download_one`函数会在多个线程中并发调用。`ThreadPoolExecutor.map()`函数返回一个生成器。

测试：

```python
>>> main(download_many)
PH BRMX  VN ET ID JP DE BD CN FR PK CD US RU EG IR IN NG TR
20 flags downloaded in 26.26s
```

## 期物

期物封装待完成的操作，可以放入队列，其完成的状态可以查询，得到结果（或抛出异常）后可以获取结果（或异常）。

`concurrent.futures.Future`类实例表示可能已经完成或者尚未完成的延迟计算。**通常情况下自己不应该创建`future`，只能由并发框架（`concurrent.futures`或`asyncio`）实例化。**原因是：期物表示终将发生的事情，而确定某件事会发生的唯一方式是执行的时间已经排定。因此，只有排定把某件事交给 `concurrent.futures.Executor` 子类处理时，才会创建 `concurrent.futures.Future` 实例。`Executor.submit()`方法的参数是一个可调用对象，调用这个方法后会为传入的可调用对象排期，并返回一个`future`。

`future`实例的几种重要的方法：

+ `done()`方法：这个方法不阻塞，返回值是布尔值，指明`future`链接的可调用对象是否已经执行；
+ `add_done_call_back()`：这个方法只有一个参数，类型是可调用的对象，`future`运行结束后会调用传入的可调用对象；
+ `result()`:在运行结束后调用该方法，返回可调用对象的结果，或者重新抛出执行可调用对象时抛出的异常。对与`concurrent.futures.Future`实例来说，调用`result()`方法会阻塞调用方所在的线程，直到有结果可返回。

`concurrent.futures.as_completed`函数的参数是一个`future`实例组成的列表，返回值是一个迭代器，在`future`运行结束后产出`future`。如果想一睹`future`真容，可使用该函数：

```python
def download_many(cc_list):
    with futures.ThreadPoolExecutor(max_workers=10) as executor:
        to_do = []
        for cc in sorted(cc_list):
            # 创建并排定future实例
            future = executor.submit(download_one, cc)
            to_do.append(future)
            print(f'Scheduled for {cc}: {future}')

        results = []

        for future in futures.as_completed(to_do):
            # 获取future实例的结果(因为future由as_completed函数产出，因此不会在该步阻塞)
            res = future.result()
            print(f'{future} result: {res}')
            results.append(res)

    return len(results)
```

测试：

```python
>>> main(download_many)
Scheduled for BD: <Future at 0x27fe59c4088 state=running>
Scheduled for BR: <Future at 0x27fe59c49c8 state=running>
Scheduled for CD: <Future at 0x27fe57ab608 state=running>
Scheduled for CN: <Future at 0x27fe5928cc8 state=running>
Scheduled for DE: <Future at 0x27fe59216c8 state=running>
Scheduled for EG: <Future at 0x27fe5926fc8 state=running>
Scheduled for ET: <Future at 0x27fe59a7148 state=running>
Scheduled for FR: <Future at 0x27fe59b8dc8 state=running>
Scheduled for ID: <Future at 0x27fe59bc8c8 state=running>
Scheduled for IN: <Future at 0x27fe582d5c8 state=running>
Scheduled for IR: <Future at 0x27fe59a17c8 state=pending>
Scheduled for JP: <Future at 0x27fe5852708 state=pending>
Scheduled for MX: <Future at 0x27fe5852648 state=pending>
Scheduled for NG: <Future at 0x27fe58523c8 state=pending>
Scheduled for PH: <Future at 0x27fe58529c8 state=pending>
Scheduled for PK: <Future at 0x27fe58528c8 state=pending>
Scheduled for RU: <Future at 0x27fe5852548 state=pending>
Scheduled for TR: <Future at 0x27fe5852348 state=pending>
Scheduled for US: <Future at 0x27fe583be88 state=pending>
Scheduled for VN: <Future at 0x27fe5852308 state=pending>
EG <Future at 0x27fe5926fc8 state=finished returned str> result: EG
CD <Future at 0x27fe57ab608 state=finished returned str> result: CD
FR <Future at 0x27fe59b8dc8 state=finished returned str> result: FR
MX <Future at 0x27fe5852648 state=finished returned str> result: MX
NG <Future at 0x27fe58523c8 state=finished returned str> result: NG
BD <Future at 0x27fe59c4088 state=finished returned str> result: BD
ET <Future at 0x27fe59a7148 state=finished returned str> result: ET
IR <Future at 0x27fe59a17c8 state=finished returned str> result: IR
BR <Future at 0x27fe59c49c8 state=finished returned str> result: BR
CN <Future at 0x27fe5928cc8 state=finished returned str> result: CN
PH <Future at 0x27fe58529c8 state=finished returned str> result: PH
IN <Future at 0x27fe582d5c8 state=finished returned str> result: IN
TR <Future at 0x27fe5852348 state=finished returned str> result: TR
ID <Future at 0x27fe59bc8c8 state=finished returned str> result: ID
US <Future at 0x27fe583be88 state=finished returned str> result: US
JP <Future at 0x27fe5852708 state=finished returned str> result: JP
PK <Future at 0x27fe58528c8 state=finished returned str> result: PK
VN <Future at 0x27fe5852308 state=finished returned str> result: VN
RU <Future at 0x27fe5852548 state=finished returned str> result: RU
DE <Future at 0x27fe59216c8 state=finished returned str> result: DE

20 flags downloaded in 14.78s
```

排定的`future`按字母表的顺序，`future`的`repr()`方法会显示`future`的状态，前10各`future`的状态时`running`，是因为我们设置了10个工作的线程。后10个`future`的状态是`pending`，表示在等待有线程可用。

`EG <Future at 0x27fe5926fc8 state=finished returned str> result: EG`中的第一个EG是在一个工作线程中的download_one函数输出的，在此之后的内容是download_many函数输出的。

CPython解释器本身就不是线程安全的，因此有全局解释器锁（Global Interpreter Lock），**一次只允许使用一个线程执行Python字节码**，因此，一个Python进程通常不能同时使用多个CPU核心。然而，标准库中所有执行阻塞型I/O操作的函数，在等待操作系统返回结果时都会释放GIL（`time.sleep()` 函数也会释放 GIL），允许其他线程运行，这意味着在Python语言这个层次上可以使用多线程。

标准库中每个使用 C 语言编写的 I/O 函数都会释放 GIL，因此，当某个线程在等待 I/O 时， Python 调度程序会切换到另一个线程。

`ProcessPoolExecutor`类把工作分配给多个python进程处理，因此，如果需要做CPU密集型处理，使用这个模块能绕开GIL，利用所有可用的CPU核心。**使用`concurrent.futures`能轻松的把基于线程的方案转换为基于进程的方案。**

基于进程的最佳进程数等于硬件中可用的所有CPU核心数，而基于线程的线程数取决于做的是什么事，以及可用的内存有多少，因此要通过仔细测试才能找到最佳线程数。

**处理I/O密集型任务时使用线程，处理CPU密集型任务时使用进程。**

加入电脑CPU有n个核心，使用进程时，程序分别将n个链接发送给n个核心，各个核心分别顺序执行访问url，获得response，保存image等操作，等该核心的任务完成后，再分配其下一个链接。而使用线程时，程序会将所有链接分配给一个cpu核心去完成，当执行到阻塞型I/O操作时（例如，等待远程服务器返回结果）或遇到`time.sleep()`语句时，立马开启另一个线程，等之前的某个线程获得response时，暂停现在正在进行的线程，转而去处理之前线程的返回结果，处理完毕后在继续刚才暂停的线程，...

---

## 实验Executor.map方法

```python
from time import sleep, strftime
from concurrent import futures
table = '\t'

def display(*args):
    print(f'{strftime("[%H:%M:%S]")}', end=' ')
    print(*args)

def loiter(n):
    display(f"{table*n} loiter({n}): doing nothing for {n}s ...")
    sleep(n)
    display(f"{table*n} loiter({n}) done!")
    return n * 10

def main():
    display('Script starting ...')
    executor = futures.ThreadPoolExecutor(max_workers=3)
    results = executor.map(loiter, range(5))
    display('result: ', results)
    display('Waiting for invidual results: ')
    for i, result in enumerate(results):
        display(f'result {i}: {result}')
```

测试：

```python
>>> main()
[17:53:46] Script starting ...
[17:53:46]  loiter(0): doing nothing for 0s ...
[17:53:46]  loiter(0) done!
[17:53:46] 	 loiter(1): doing nothing for 1s ...
[17:53:46] 		 loiter(2): doing nothing for 2s ...
[17:53:46] result:  <generator object Executor.map.<locals>.result_iterator at 0x0000027FE5973448>
[17:53:46] Waiting for invidual results: 
[17:53:46] 			 loiter(3): doing nothing for 3s ...
[17:53:46] result 0: 0
[17:53:47] 	 loiter(1) done!
[17:53:47] 				 loiter(4): doing nothing for 4s ...
[17:53:47] result 1: 10
[17:53:48] 		 loiter(2) done!
[17:53:48] result 2: 20
[17:53:49] 			 loiter(3) done!
[17:53:49] result 3: 30
[17:53:51] 				 loiter(4) done!
[17:53:51] result 4: 40
```

设置了`max_worker=3`，因此同时最多可以分配3个线程，程序一开始分配三个线程：`loiter(0)`,`ioter(1)`和`lioter(2)`。由于`lioter(0)`sleep了0秒，因此，该线程立马结束，于是空出了一个线程，并立马启动了下一个等待的线程`lioter(3)`【这一点可以从`lioter(0)`,`lioter(1)`,`lioter(2)`和`lioter(3)`的启动时间相同，均为`[17:53:46]`可以看出】

`Executor.map`函数有个特性：返回结果的顺序与调用开始的顺序一致。如果第一个调用从开始到生成结果总共用时10秒，而其它调用只用1秒，那么代码会阻塞10秒，以获取map方法返回的生成器产出的第一个结果，在此时候，获取后续结果时不会阻塞，因为后续的调用已经结束。
