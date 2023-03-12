# import tensorflow as tf

# def get_center_loss(features, labels, alpha, num_classes, name):

#     len_features = features.get_shape()[1]
  
#     centers = tf.get_variable(name, [num_classes, len_features], dtype=tf.float32,
#                               initializer=tf.constant_initializer(0), trainable=False)
    
#     labels = tf.reshape(labels, [-1])
#     centers_batch = tf.gather(centers, labels)
#     diff = centers_batch - features
#     unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
#   appear_times = tf.gather(unique_count, unique_idx)
#     appear_times = tf.reshape(appear_times, [-1, 1])

#     diff = diff / tf.cast((1 + appear_times), tf.float32)
#     diff = alpha * diff

#     centers_update_op = tf.scatter_sub(centers, labels, diff)

#     with tf.control_dependencies([centers_update_op]):
#         loss = tf.reduce_mean(tf.abs(features-centers_batch))
#     return loss, centers







import torch

def get_center_loss(features, labels, alpha, num_classes, name):

    len_features = features.size(1)

    centers = torch.zeros(num_classes, len_features).float()
    if torch.cuda.is_available():
        centers = centers.cuda()

    labels = labels.view(-1, 1).long()
    centers_batch = centers.index_select(0, labels.squeeze()) # gather -> index_select
    diff = centers_batch - features
    unique_label, unique_idx, unique_count = torch.unique(labels, sorted=True, return_inverse=True, return_counts=True)
    appear_times = unique_count.index_select(0, unique_idx)
    appear_times = appear_times.view(-1, 1)

    diff = diff / (1 + appear_times.float())
    diff = alpha * diff

    centers_update_op = torch.zeros(num_classes, len_features).scatter_add_(0, labels.repeat(1, len_features), diff)
    # scatter_sub -> scatter_add_

    loss = torch.mean(torch.abs(features - centers_batch))

    return loss, centers

output, inverse_indices, counts = torch.unique(torch.tensor([[1, 3], [2, 3]], dtype=torch.long), sorted=True, return_inverse=True)
print(output)
print(inverse_indices)
print(counts)