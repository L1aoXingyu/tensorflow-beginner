## TensorBoard 可视化
tensorflow不仅仅是一个计算图软件，其还包含了tensorboard可视化工具，安装tensorflow的时候会默认安装，使用方法非常简单，使用`writer = tf.summary.FileWriter('./graph', sess.graph)`就能够创建一个文件写入器，`./graph`是存储目录，`sess.graph`表示读入的图结构。

我们可以写一个简单的小程序
```python
import tensorflow as tf
a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(x))
writer.close()  # close the writer when you’re done using it

```

然后打开终端，运行程序，接着输入`tensorboard --logdir="./graphs"`，然后打开网页输入 http://localhost:6006/，就能够进入tensorboard，可以得到下面的结果。

![screenshot.png](http://upload-images.jianshu.io/upload_images/3623720-5989662b943015bb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 常数类型(Constant types)
能够通过下面这个方式创造一个常数

```python
tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)
```

比如建立一维向量和矩阵，然后将他们乘起来

```python
a = tf.constant([2, 2], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')
x = tf.multiply(a, b, name='dot_production')
with tf.Session() as sess:
    print(sess.run(x))
>> [[0, 2]
    [4, 6]]
```

这跟numpy里面的是差不多的，同时还有一些特殊值的常量创建。

```python
tf.zeros(shape, dtype=tf.float32, name=None)
tf.zeros_like(input_tensor, dtype=None, name=None, optimize=True)
tf.ones(shape, dtype=tf.float32, name=None)
tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)
tf.fill(dims, value, name=None)
tf.fill([2, 3], 8)
>> [[8, 8, 8], [8, 8, 8]]
```

也有和numpy类似的序列创建

```python
tf.linspace(start, stop, num, name=None)
tf.linspace(10.0, 13.0, 4)
>> [10.0, 11.0, 12.0, 13.0]
tf.range(start, limit=None, delta=1, dtype=None, name='range')
tf.range(3, limit=18, delta=3)
>> [3, 6, 9, 12, 15]
```

这和numpy最大的区别在于其不能迭代，即

```python
for _ in tf.range(4): # TypeError
```

除此之外还可以产生一些随机数

```python
tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None,
name=None)
tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None,
name=None)
tf.random_shuffle(value, seed=None, name=None)
tf.random_crop(value, size, seed=None, name=None)
tf.multinomial(logits, num_samples, seed=None, name=None)
tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)
```

另外tensorflow和numpy的数据类型可以通用，也就是说
```python
tf.ones([2, 2], np.float32)
>> [[1.0, 1.0], [1.0, 1.0]]
```

最好不要使用python自带的数据类型，同时在使用numpy数据类型的时候要小心，因为未来可能tensorflow的数据类型和numpy不再兼容。

## 变量(Variable)
使用常量会存在什么问题呢？常量会存在计算图的定义当中，如果常量过多，这会使得加载计算图变得非常慢，同时常量的值不可改变，所以引入了变量。

```python
a = tf.Variable(2, name='scalar')
b = tf.Variable([2, 3], name='vector')
c = tf.Variable([[0, 1], [2, 3]], name='matrix')
w = tf.Variable(tf.zeros([784, 10]), name='weight')
```

变量有着下面几个操作

```python
x = tf.Variable()
x.initializer # 初始化
x.eval() # 读取里面的值
x.assign() # 分配值给这个变量
```

注意一点，在使用变量之前必须对其进行初始化，初始化可以看作是一种变量的分配值操作。最简单的初始化方式是一次性初始化所有的变量

```python
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```

也可以对某一部分变量进行初始化

```python
init_ab = tf.variable_initializer([a, b], name='init_ab')
with tf.Session() as sess:
    sess.run(init_ab)
```

或者是对某一个变量进行初始化

```python
w = tf.Variable(tf.zeros([784, 10]))
with tf.Session() as sess:
    sess.run(w.initializer)
```

如果我们想取出变量的值，有两种方法

```python
w = tf.Variable(tf.truncated_normal([10, 10], name='normal'))
with tf.Session() as sess:
    sess.run(w.initializer)
    print(w.eval()) # 方法一
    print(sess.run(w)) # 方法二
```

下面看看这个小程序

```python
w = tf.Variable(10)
w.assign(100)
with tf.Session() as sess:
    sess.run(w.initializer)
    print(w.eval())
>> 10
```

上面这个程度会得到10,这是因为我们虽然定义了assign操作，但是tensorflow是在session中执行操作，所以我们需要执行assign操作。

```python
w = tf.Variable(10)
assign_op = w.assign(100)
with tf.Session() as sess:
    sess.run(w.initializer)
    sess.run(assign_op)
    print(w.eval())
>> 100
```

另外tensorflow的每个session是相互独立的，我们可以看看下面这个例子

```python
W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()
sess1.run(W.initializer)
sess2.run(W.initializer)
print(sess1.run(W.assign_add(10))) # >> 20
print(sess2.run(W.assign_sub(2))) # >> 8
print(sess1.run(W.assign_add(100))) # >> 120
print(sess2.run(W.assign_sub(50))) # >> -42
sess1.close()
sess2.close()
```

你也可以根据一个变量来定义一个变量

```python
w = tf.Variable(tf.truncated_normal([700, 10]))
u = tf.Variable(w * 2)
```

## 占位符(Placeholders)
tensorflow中一般有两步，第一步是定义图，第二步是在session中进行图中的计算。对于图中我们暂时不知道值的量，我们可以定义为占位符，之后再用`feed_dict`去赋值。

定义占位符的方式非常简单

```python
tf.placeholder(dtype, shape=None, name=None)
```

dtype是必须要指定的参数，shape如果是None，说明任何大小的tensor都能够接受，使用shape=None很容易定义好图，但是在debug的时候这将成为噩梦，所以最好是指定好shape。

我们可以给出下面的小例子。

```python
a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([5, 5, 5], tf.float32)
c = a + b
with tf.Session() as sess:
    print(sess.run(c, feed_dict={a: [1, 2, 3]}))
```

除此之外，也可以给tensorflow中的运算进行feed操作，如下

```python
a = tf.add(2, 3)
b = tf.multiply(a, 3)
with tf.Session() as sess:
    print(sess.run(b, feed_dict={a: 2}))
>> 6
```

## lazy loading
lazy loading是指你推迟变量的创建直到你必须要使用他的时候。下面我们看看一般的loading和lazy loading的区别。

```python
# normal loading
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        sess.run(z)

# lazy loading
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        sess.run(tf.add(x, y))
```

normal loading 会在图中创建x和y变量，同时创建x+y的运算，而lazy loading只会创建x和y两个变量。这不是一个bug，那么问题在哪里呢？

normal loading在session中不管做多少次x+y，只需要执行z定义的加法操作就可以了，而lazy loading在session中每进行一次x+y，就会在图中创建一个加法操作，如果进行1000次x+y的运算，normal loading的计算图没有任何变化，而lazy loading的计算图会多1000个节点，每个节点都表示x+y的操作。

看到了吗，这就是lazy loading造成的问题，这会严重影响图的读入速度。
