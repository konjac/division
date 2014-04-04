
## 基本情况

$$
\begin{align}
\dfrac{1}{1+y} &=\dfrac{1-y}{1-y^2} \\
&=\dfrac{(1-y)(1+y^2)}{1-y^4}\\
&=\cdots\\
&=\dfrac{(1-y)(1+y^2)\cdots (1+y^{2^k})\cdots}{1-y^{2^{k+1}}} \\
&\approx (1-y)(1+y^2)\cdots (1+y^{2^k})\cdots
\end{align}
$$

## 查表法

$$
\begin{align}
\dfrac{1}{x} &= \dfrac{2^k}{x\cdot2^k} &&\\
&=\dfrac{2^k}{\lfloor x\cdot2^k\rfloor + x_0} &&\\
&=\dfrac{2^k}{n + x_0}&&n = \lfloor x\cdot2^k\rfloor, x0 = x\cdot 2^k - n \\
&=\dfrac{\dfrac{2^k}{n}}{1+\dfrac{x_0}{n}} \\
&=\dfrac{\dfrac{2^k}{n}}{1+y}&& y = \dfrac{x_0}{n} = \dfrac{x\cdot 2^k - n}{n} = \dfrac{x\cdot 2^k}{n}-1\\
&(\dfrac{1}{n} \text{or} \dfrac{2^k}{n}  \text{can be preprocessed})
\end{align}
$$