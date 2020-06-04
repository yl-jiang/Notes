## str

### 批量替换字符

``` python
multi_map = str.maketrans({'a':'A', 'b': 'B'})

'asdadavbkjndnbasdao'.translate(multi_map)
```

输出：

```python
'AsdAdAvBkjndnBAsdAo'
```

## 规范化Unicode字符串

因为 `Unicode` 有组合字符（变音符号和附加到前一个字符上的记号，打印时作为一个整体），所以字符串比较起来很复杂。肉眼看上去相同的字符，在内部表示上可能用的是不同的码位，比较时Python就会判定二者不相同。为了避免这一情况的发生，在进行字符串(**特别是含有非ASCII码的字符串**)的比较之前最好对其进行标准化。

``` python
from unicodedata import normalize

s1 = 'café'
s2 = 'cafe\u0301'

# s1和s2是两个相同字符串的不同码位表示
s1 == s2 # outpot: False
normailze('NFC', s1) == normalize('NFC', s2) # output: True
```

西方键盘通常能输出组合字符，因此用户输入的文本默认是 `NFC` 形式。不过，安全起见，保存文本之前，最好使用 `normalize('NFC', user_text)` 清洗字符串.

## str.lower()和str.casefold()

对字符串进行不区分大小写的比较应该尽可能的使用`str.casefold()`。因为`str.lower()`会无法处理两个例外情况(`\mu`的大小写，以及德语的 'Eszett')
