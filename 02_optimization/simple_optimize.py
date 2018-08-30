import tensorflow as tf

# x placeholder for our training data
x = tf.placeholder(tf.float64)

# a is the variable storing our values. It is initialised with starting "guess"
# the value of this variable is the "a" in our equation
a = tf.Variable(10, name="w", dtype=tf.float64)

# Our model of y = x + a
operation = x + a

# the loss defines how wrong our model is, if the loss is 0 - ther is no loss - the optimum is reached
# loss = sqrt(x - y) -> loss = sqrt(x - x + a)
loss = tf.square(x - operation)

# The optimizer is trying to optimize the loss function over the training data
# x is the training data, a is the variable value(initialized with a starting guessing value) which will be adopted
# What the optimizer solves is : What is the optimal value of "a" so that loss
# loss = sqrt(x - x + a) will become as low as possible
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# Normal TensorFlow - initialize values, create a session and run the model
model = tf.global_variables_initializer()

# learn / train the result variable(a).We humans are intelligent and most of us learned math at shool and 
# we can abstract(pattern recognition) / simulate(make prediction based on learned data) - so we can already predict that a should become 0
# as srqt(x- x + 0) = -> sqrt(0 + 0) -> 0, but let the model learn this value for the training data and adopt the variable value.
# In the end - the variable value should be 0 (will never reach but will become very very close). Variable A was learned/optimized
# Take a look for value of a at every iteration!
with tf.Session() as session:
    session.run(model)
    for i in range(1600):
        x_value = i
        print("x:", x_value)
        session.run(train_op, feed_dict={x: x_value})
        a_value = session.run(a)
        print("Optimal variable value: ", a_value)

