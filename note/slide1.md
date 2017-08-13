## 介绍
深度学习后面的数学概念已经存在10多年，但是深度学习框架是最近几年才出来的。现在大量的框架在灵活性上和便于工业界使用上做了权衡，灵活性对于科研非常重要，但是对于工业界太慢了，但是换句话说，足够快，能够用于分布式的框架只适用于特别的网络结构，这对科研又不够灵活。这留给了使用者一个矛盾的境地：我们是否应该尝试用没有那么灵活的框架做科研，这样当应用于工业界的时候，我们不必再重新用另外一个框架复现代码；或者是我们是否应该在做研究的时候使用一个框架，在工业界应用的时候使用另外一个完全不同的框架呢？

如果选择前者，那么做研究的时候并不方便尝试很多不同类型的网络，如果选择后者，我们必须要重新复现代码，这容易导致实验结果和工业应用上不同，我们也需要付出很多精力去学习。

TensorFlow的出现希望解决这个矛盾的事情。

## 什么是TensorFlow？
- 使用数据流和图来做数值计算的开源软件，用于机器智能

- 主要是由Google Brain团队开发用于机器学习和深度神经网络的研究

- 能够应用于广泛的领域

虽然TensorFlow是开源的，但是只有GitHub上的部分是开源的，Google还有一个内部版本，官方说法是Google的内部版本有很多转为其定制的工具和服务，大众没有需求使用，并不是Google的开源没有诚意，希望如此吧。

## 为什么使用TensorFlow？
- Python API，这是大多数深度学习框架都有的

- 能够使用多个CPU和GPU，最重要的是能够很容易部署到服务器上和移动端，这是很多框架不能做的事

- 足够灵活，非常低层

- tensorboard可视化非常好

- Checkpoints作为实验管理，能够随时保存模型

- 自动微分

- 庞大的社区

- 大量优秀的项目正在使用TensorFlow

## Getting Started

### tensor
0-d tensor：标量，1-d tensor：向量，2-d tensor：矩阵

### 数据流图

![screenshot.png](http://upload-images.jianshu.io/upload_images/3623720-410267585c558052.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```python
import tensorflow as tf
a = tf.add(3, 5)
print(a)
>> Tensor("Add: 0", shape=(), dtype=int32)
```

并不能得到8，需要开启session，在session中操作能够被执行，Tensor能够被计算，这点有点反人类，跟一般的推断式编程是不同的，比如PyTorch

```python
import tensorflow as tf
a = tf.add(3, 5)
sess = tf.Session()
print(sess.run(a))
sess.close()
>> 8
```

当然可以使用一种更高效的写法

```python
import tensorflow as tf
a = tf.add(3, 5)
with tf.Session() as sess:
    print(sess.run(a))
```

当然可以建立更复杂的计算图如下

```python
x = 2
y = 3
add_op = tf.add(x, y)
mul_op = tf.mul(x, y)
uesless = tfmul(x, add_op)
pow_op = tf.pow(add_op, mul_op)
with tf.Session() as sess:
    z, not_useless = sess.run([op3, useless])
```

在`sess.run`调用的时候使用[]来得到多个结果。

也可以将图分成很多小块，让他们在多个CPU和GPU下并行

![screenshot.png](http://upload-images.jianshu.io/upload_images/3623720-5babbbcce36260b9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以将计算图的一部分放在特定的GPU或者CPU下

```python
with tf.device('/gpu:2'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='b')
    c = tf.matmul(a, b)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))
```

尽量不要使用多个计算图，因为每个计算图需要一个session，而每个session会使用所有的显卡资源，必须要用python/numpy才能在两个图之间传递数据，最好在一个图中建立两个不联通的子图

## 为什么使用Graph
1. 节约计算资源，每次运算仅仅只需运行与结果有关的子图

2. 可以将图分成小块进行自动微分

3. 方便部署在多个设备上

4. 很多机器学习算法都能够被可视化为图的结构