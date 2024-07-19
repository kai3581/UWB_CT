import tensorflow as tf

datatensor = tf.constant([([1],[2]),([3],[4])])
dataset = tf.data.Dataset.from_tensor_slices(datatensor)
for element in dataset:
    tf.print(element[0])
    tf.print(element[1])

batched_dataset = dataset.batch(2)
for batch in batched_dataset.take(1):
    tf.print(batch[:,0])

for batch in dataset.batch(1):
    tf.print(batch)

dtensor = tf.constant([1, 2, 3, 4])

dd = tf.data.Dataset.from_tensor_slices(dtensor)

for e in dd:
    tf.print(e)

for batch in dd.batch(1):
    tf.print(batch)

ttt = tf.constant([([[1]], [[2]]), ([[3]], [[4]]), ([[5]], [[6]])])
tf.print(ttt)
ttt = tf.unstack(ttt, axis=1)
tf.print(ttt)
