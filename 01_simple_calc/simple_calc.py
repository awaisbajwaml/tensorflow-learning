import tensorflow as tf

x_value = 5

# x placeholder for our input data
x = tf.placeholder(tf.float64)

# Our model of y = x + x
operation = x + x

# notice that you build a graph of mathematical operations
# it does not performan any computation until you run it
print(x)
print(operation)

with tf.Session() as session:
    #compute 5+ 5
    result = session.run(operation, feed_dict={x: x_value})
    print("result: ", result)

