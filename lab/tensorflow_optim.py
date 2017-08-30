import tensorflow as tf

x = tf.Variable(5, dtype=tf.float32)
y = tf.Variable(3, dtype=tf.float32)

z = x**2 + x * y + 3

sess = tf.Session()
# initialize variable
sess.run(tf.global_variables_initializer())

# define optimizer
optimizer = tf.train.GradientDescentOptimizer(0.1)

# compute gradient z w.r.t x and y
grads_and_vars = optimizer.compute_gradients(z, [x, y])

# fetch the variable
get_grads_and_vars = [(gv[0], gv[1]) for gv in grads_and_vars]

# dz/dx = 2*x + y= 13
# dz/dy = x = 5
print('grads and variables')
print('x: grad {}, value {}'.format(
    sess.run(get_grads_and_vars[0][0]), sess.run(get_grads_and_vars[0][1])))

print('y: grad {}, value {}'.format(
    sess.run(get_grads_and_vars[1][0]), sess.run(get_grads_and_vars[1][1])))

print('Before optimization')
print('x: {}, y: {}'.format(sess.run(x), sess.run(y)))

# optimize parameters
opt = optimizer.apply_gradients(get_grads_and_vars)
# x = x - 0.1 * dz/dx = 5 - 0.1 * 13 = 3.7
# y = y - 0.1 * dz/dy = 3 - 0.1 * 5 = 2.5
print('After optimization using learning rate 0.1')
sess.run(opt)
print('x: {:.3f}, y: {:.3f}'.format(sess.run(x), sess.run(y)))
sess.close()