
# 拓展算法

对于$\dfrac{1}{x}$如果能够快速地取得一个近似值$a$,$ax\approx 1$,可以令$y=ax-1$,只要$y$很小,那么做如下变化就可以使用$\dfrac{1}{1+y}$进行展开了

$$
\dfrac{1}{x}=\dfrac{a}{ax} = \dfrac{a}{1+y}
$$

**查表法其实就是用取整、查表的方法来估计$a$,但是在MIC上查表的效率不尽人意.**

获取近似值$a$,有以下的一些方法

### 一次多项式

记$a=f(1+y)$,多项式逼近的思路是在$[0,1]$上获得一个多项式估计使得$max|(1+y)*f(y)-1|$最小.然后即可对任意的$x$取出尾数部分作为$1+y$计算,最后把指数部分添加上去即可.

$$
\begin{align}
&f(x)= -\dfrac{8}{17}x+\dfrac{16}{17} &&error = \dfrac{1}{17}
\end{align}
$$
### 二次多项式

$$
\begin{align}
f(x)=&\dfrac{245}{796}x^2 - \dfrac{935}{1194}x + \dfrac{2345}{2388} & error =&  \dfrac{43}{2388}= \dfrac{1}{55.534\cdots} \\
f(x)=&\dfrac{32}{99}x^2 - \dfrac{80}{99}x + \dfrac{98}{99} & error =& \dfrac{1}{99}
\end{align}
$$

### 三次多项式

$$
\begin{align}
f(x)=&-\dfrac{642}{3035}x^3+\dfrac{1954}{3035}x^2-\dfrac{8468}{9105}x+\dfrac{63508}{63735} &error =& 0.00141655 \\
f(x)=&-0.2115 x^3+0.6438 x^2-0.93 x+0.9964&error=&0.00137882 \\
\end{align}
$$

### 类似卡马克快速平方根倒数的方法

IEEE754的双精度浮点数可以表示为 $2^e(1+m)$.对$\dfrac{1}{x}=y$等号左右取底为2的对数

$$
\begin{align}
\dfrac{1}{x}&=y \\
-\log x&=log y \\
-\log (2^{e_x}(1+m_x)) &= \log (2^{e_y}(1+m_y)) \\
-e_x-\log(1+m_x) &= e_y+\log(1+m_y) \\
\end{align}
$$

由于$m\in [0,1)$, 所以可以做一个近似$\log(1+m)=m+\Delta$.
$$
\begin{align}
-e_x-\log(1+m_x) &= e_y+\log(1+m_y) \\
-(e_x+m_x+\Delta_x) &= e_y+m_y + \Delta_y
\end{align}
$$

如果把浮点数看做是64位整数$2^{52}E+M$, 有如下的对应关系

$$
\begin{align}
e&=E-1023\\
m&=\dfrac{M}{2^{52}}
\end{align}
$$

于是又得到

$$
\begin{align}
&-(e_x+m_x+\Delta_x） &&= e_y+m_y + \Delta_y \\
&-(E_x-1023+\dfrac{M_x}{2^{52}}+\Delta_x） &&=  (E_y-1023+\dfrac{M_y}{2^{52}})+\Delta_y \\
&-(2^{52}E_x-2^{52}\cdot 1023+M_x+2^{52}\Delta_x) &&=  (2^{52}E_y-2^{52}\cdot 1023+M_y)+2^{52}\Delta_y \\
&-(2^{52}E_x+M_x)+2^{52}(1023-\Delta_x) &&=  (2^{52}E_y+M_y)+2^{52}(\Delta_y-1023) \\
\end{align}
$$

$2^{52}E_x+M_x$和$2^{52}E_y+M_y$正好是$x$和$y$对应的64位整数, 整理一下

$$
\begin{align}
(2^{52}E_y+M_y) &= -(2^{52}E_x+M_x)+2^{52}(1023-\Delta_x) -2^{52}(\Delta_y-1023) \\
&= 2^{52}( 2046-(\Delta_x +\Delta_y))-(2^{52}E_x+M_x) \\
y_{int64} &= 2^{52}( 2046-(\Delta_x +\Delta_y))-x_{int64}
\end{align}
$$

把$\Delta_x +\Delta_y$使用一个常数进行近似.由于$\Delta= log(1+m)-m \in [0, 0.0860...], m\in[0,1]$, 于是可以用$\Delta_{max}$来近似表示$\Delta_x +\Delta_y$, 这样, 误差为

$$
\epsilon=\Delta_x +\Delta_y-\Delta_{max} \in [-\Delta_{max}, \Delta_{max}]
$$

于是算得所谓MAGIC NUMBER

$$
2^{52}( 2046-\Delta_{max}) =0x7fde9f73aabb2400
$$

所以得到了快速的倒数近似

    union {
        long long i;
        double y;
    } p;
    p.y = x;
    p.i = 0x7fde5f73aabb2400 - p.i;
    rec = p.y;

使用暴力验证的方法可以算得这样做出来的误差$ax-1$大约是$0.05084$,精度虽然不很高,但是仅仅需要一条指令即可完成估计,且不需要单独处理尾数(拆分尾数和合并指数至少需要4条指令), 可以让$\dfrac{1}{1+y}$多迭代1次.

<!--
| 日期 | 时间(北京时间) | 时长|  | |
| --------: | :--------| :--: |-----|----|
|     3.12 周三      | 3:00am     | 32天5小时| Registration Opens       |   |
|	 4.12 周六      | 7:00am			 | 25小时|  Qualification Round Starts| 比赛时会划定晋级分数线 |
|	 4.13 周日     |  8:00am             |  | Qualification Round Ends, Registration Closes          | |
|	 4.26 周六    |   9:00am – 11:30am    | 2.5小时| Online Round 1: Sub-Round A          | 前1000名晋级Round2|
|	 5.4 周日    |    12:00am – 2:30am    | 2.5小时| Online Round 1: Sub-Round B          | 前1000名晋级Round2|
|	 5.11 周日  |     5:00pm – 7:30pm     | 2.5小时| Online Round 1: Sub-Round C          | 前1000名晋级Round2 |
|	 5.31 周六      | 10:00pm – 12:30am   | 2.5小时| Online Round 2  | 前1000名获得T-shirt,前500名晋级Round3 |      |
|	 6.14 周六    |   10:00pm – 12:30am   | 2.5小时| Online Round 3| 2013年的冠军和除此之外的前25名晋级Onsite|
|    8.15            |                        |        | Onsite(at the Google offices in LA, USA)  |     年满18岁才能进Onsite |



|名次	|奖金|
|:-----:|-----|
|冠军	|\$15,000 USD|
|亚军	|\$2,000 USD|
|季军	|\$1,000 USD|
|4—26名	|\$100 USD|
|R2的前1000 | T-shirt|

-->
