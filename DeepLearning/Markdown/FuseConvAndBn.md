卷积层操作：
$$
y_{5}=\sum^{n=9}_{i=1}{x_{i}\cdot{c_{i}}}+b_{conv} \tag 1 \\
$$
Batchnorm层操作：
$$
\begin{align*}
z_{5} &= \frac{y_{5}-m_{bn}}{\sqrt{\delta + \epsilon}}\cdot{\gamma}+\beta \\
&=\frac{\gamma}{\sqrt{\delta + \epsilon}}\cdot{y_5}+\beta - \frac{\gamma\cdot{m_{bn}}}{\sqrt{\delta + \epsilon}} \\
&=\frac{\gamma}{\sqrt{\delta + \epsilon}}\cdot{(\sum^{n=9}_{i=1}{x_{i}\cdot{c_{i}}}+b_{conv})}+\beta - \frac{\gamma\cdot{m_{bn}}}{\sqrt{\delta + \epsilon}}\tag 2  \\ 
&=\frac{\gamma}{\sqrt{\delta + \epsilon}}\cdot{\sum^{n=9}_{i=1}x_i c_i} + (\frac{\gamma}{\sqrt{\delta + \epsilon}}\cdot{b_{conv}}+\beta - \frac{\gamma\cdot{m_{bn}}}{\sqrt{\delta + \epsilon}})   \\
&=\sum^{n=9}_{i=1}x_i(\frac{\gamma}{\sqrt{\delta + \epsilon}}\cdot{c_i})+(\frac{\gamma}{\sqrt{\delta + \epsilon}}\cdot{b_{conv}}+\beta - \frac{\gamma\cdot{m_{bn}}}{\sqrt{\delta + \epsilon}})
\end{align*}
$$
$z_5$就是经过卷积和Batchnorm层操作后的结果，将上述两个操作可以合二为一，变成一个卷积层，其中卷积核参数为$\frac{\gamma}{\sqrt{\delta + \epsilon}}\cdot{c_i}$，卷积层bias为$\frac{\gamma}{\sqrt{\delta + \epsilon}}\cdot{b_{conv}}+\beta - \frac{\gamma\cdot{m_{bn}}}{\sqrt{\delta + \epsilon}}$

