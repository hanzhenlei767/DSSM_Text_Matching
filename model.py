import tensorflow as tf
import numpy as np
import random
import json
import os
import logging
import sys
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix,accuracy_score


log_dir = 'logdir'
log_file = os.path.join(log_dir, 'log.txt')
if not os.path.isdir(log_dir):
  os.makedirs(log_dir)

logger = logging.getLogger()
logger.addHandler(logging.FileHandler(log_file))
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

model_path = "./checkpoint"
if not os.path.isdir(model_path):
  os.makedirs(model_path)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 文件路径
path = 'data/'
original_path = path+'original_data/'
assist_path = path+'assist_data/'
mid_path = path+'mid_data/'

train_result_ids = np.load(mid_path+"train_result_ids.npy")
train_fw1_ids = np.load(mid_path+"train_fw1_ids.npy")
train_fw2_ids = np.load(mid_path+"train_fw2_ids.npy")
train_labels = np.load(mid_path+"train_labels.npy").astype(np.float32)

dev_result_ids = np.load(mid_path+"dev_result_ids.npy")
dev_fw1_ids = np.load(mid_path+"dev_fw1_ids.npy")
dev_fw2_ids = np.load(mid_path+"dev_fw2_ids.npy")
dev_labels = np.load(mid_path+"dev_labels.npy").astype(np.float32)

test_result_ids = np.load(mid_path+"test_result_ids.npy")
test_fw1_ids = np.load(mid_path+"test_fw1_ids.npy")
test_fw2_ids = np.load(mid_path+"test_fw2_ids.npy")
test_labels = np.load(mid_path+"test_labels.npy").astype(np.float32)

#word2id=json.load(open(mid_path+"word2id.json",encoding='utf-8'))
#embedding_matrix = np.random.randn(len(word2id),300).astype(np.float32)
embedding_matrix = np.load(mid_path+"embedding_matrix.npy").astype(np.float32)

logging.info(train_result_ids.shape)
logging.info(dev_result_ids.shape)
logging.info(test_result_ids.shape)

logging.info(embedding_matrix.shape)

"""
Config
"""

x1_max_len = 200
x2_max_len = 200
x3_max_len = 200
embedding_size = 300

class_nums = 5

epoch_num = 100000
batch_size = 64
lr = 0.001
clip = 5

weight = [30,9,2,2,9]

#weight = [1,1,1,1,1]
isTrain = False

"""
Model
"""
graph = tf.Graph()
with graph.as_default():
  with tf.variable_scope('placeholder'):
    X1 = tf.placeholder(tf.int32, name='X1',shape=(None, x1_max_len))
    X2 = tf.placeholder(tf.int32, name='X2',shape=(None, x2_max_len))
    X3 = tf.placeholder(tf.int32, name='X3',shape=(None, x3_max_len))
    labels = tf.placeholder(tf.float32, name='Y', shape=(None,class_nums))
    dropout = tf.placeholder(tf.float32, shape=(),name='dropout')
  with tf.variable_scope('embedding'):
    embedding = tf.get_variable('embedding',initializer = embedding_matrix,dtype=tf.float32, trainable=True)

    embed1 = tf.nn.embedding_lookup(embedding, X1)
    embed2 = tf.nn.embedding_lookup(embedding, X2)
    embed3 = tf.nn.embedding_lookup(embedding, X3)

    embed1 = tf.nn.dropout(embed1, dropout)
    embed2 = tf.nn.dropout(embed2, dropout)
    embed3 = tf.nn.dropout(embed3, dropout)

  with tf.variable_scope('Attention'):
    #在词Embeding层上添加attention层
    attention_w = tf.get_variable('attention_omega', [embedding_size, 1])
    attention_b = tf.get_variable('attention_b', [1,x1_max_len])

    attention1 = tf.reduce_sum(embed1 * tf.expand_dims(tf.nn.softmax(tf.tanh(tf.add(
        tf.reshape(tf.matmul(tf.reshape(embed1, [-1, embedding_size]), attention_w),[-1, x1_max_len]),
        attention_b))), -1),axis = 1)

    attention2 = tf.reduce_sum(embed2 * tf.expand_dims(tf.nn.softmax(tf.tanh(tf.add(
        tf.reshape(tf.matmul(tf.reshape(embed2, [-1, embedding_size]), attention_w),[-1, x2_max_len]),
        attention_b))), -1),axis = 1)

    attention3 = tf.reduce_sum(embed3 * tf.expand_dims(tf.nn.softmax(tf.tanh(tf.add(
        tf.reshape(tf.matmul(tf.reshape(embed3, [-1, embedding_size]), attention_w),[-1, x3_max_len]),
        attention_b))), -1),axis = 1)

  with tf.variable_scope('one_layer_word_bi-lstm'):   
    #词1层bi-lstm
    cell_fw = tf.nn.rnn_cell.LSTMCell(50)
    cell_bw = tf.nn.rnn_cell.LSTMCell(50)
    (lstm1_output_fw_seq, lstm1_output_bw_seq), states1 = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
        embed1,dtype=tf.float32)
    lstm1_bi_output = tf.concat([states1[0].h, states1[1].h], axis=-1)

    (lstm2_output_fw_seq, lstm2_output_bw_seq), states2 = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
        embed2,dtype=tf.float32)
    lstm2_bi_output = tf.concat([states2[0].h, states2[1].h], axis=-1)

    (lstm3_output_fw_seq, lstm3_output_bw_seq), states3 = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
        embed3,dtype=tf.float32)
    lstm3_bi_output = tf.concat([states3[0].h, states3[1].h], axis=-1)

  with tf.variable_scope('two_layer_lstm'): 
    #词2层单向lstm
    cell_1 = tf.nn.rnn_cell.LSTMCell(50)
    cell_2 = tf.nn.rnn_cell.LSTMCell(50)
    output,state=tf.nn.dynamic_rnn(cell=cell_1,inputs=embed1,dtype=tf.float32)
    #第一层的output输出
    lstm1_output = output

    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([cell_1,cell_2])


    output,state=tf.nn.dynamic_rnn(cell=lstm_cell,inputs=embed1,dtype=tf.float32)

    #第二层的h输出
    lstm1_2_output = state[1].h
    
    output,state=tf.nn.dynamic_rnn(cell=cell_1,inputs=embed2,dtype=tf.float32)
    #第一层的output输出
    lstm2_output = output
    
    output,state=tf.nn.dynamic_rnn(cell=lstm_cell,inputs=embed2,dtype=tf.float32)
    #第二层的h输出
    lstm2_2_output = state[1].h 

    output,state=tf.nn.dynamic_rnn(cell=cell_1,inputs=embed3,dtype=tf.float32)
    #第一层的output输出
    lstm3_output = output
    
    output,state=tf.nn.dynamic_rnn(cell=lstm_cell,inputs=embed3,dtype=tf.float32)
    #第二层的h输出
    lstm3_2_output = state[1].h 

  with tf.variable_scope("cnn"):
    #定义权值
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, mean=0,stddev=0.1)
        return tf.Variable(initial)
    #定义偏置
    def bias_variable(shape):
        initial = tf.constant(0.1,shape=shape)
        return tf.Variable(initial)
    #定义卷积
    def conv2d(x,W):
        #srtide[1,x_movement,y_,movement,1]步长参数说明
        return tf.nn.conv2d(x,W,strides = [1,1,1,1],padding='SAME')
        #x为输入，W为卷积参数，[5,5,1,32]表示5*5的卷积核，1个channel，32个卷积核。strides表示模板移动步长，
        #SAME和VALID两种形式的padding，valid抽取出来的是在原始图片直接抽取，结果比原始图像小，
        #same为原始图像补零后抽取，结果与原始图像大小相同。
    #定义pooling
    def max_pool_3x3(x):
        #ksize   strides
        return tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,3,3,1],padding='VALID')

    cnn_input1 = tf.matmul(lstm1_output,tf.transpose(lstm2_output,[0,2,1]))#矩阵乘法
    cnn_input2 = tf.matmul(lstm1_output,tf.transpose(lstm3_output,[0,2,1]))#矩阵乘法
    cnn_input = tf.concat([cnn_input1, cnn_input2],1)
    logging.info(cnn_input)#[?,400,200]

    cnn_input = tf.reshape(cnn_input,[-1,400,200,1])
    ##卷积层conv1
    W_conv1 = weight_variable([3,3,1,8])#第一层卷积：卷积核大小3x3,1个颜色通道，8个卷积核
    b_conv1 = bias_variable([8])#第一层偏置
    h_conv1 = tf.nn.relu(conv2d(cnn_input,W_conv1)+b_conv1)#第一层输出：输出的非线性处理28x28x32
    logging.info(h_conv1)
    h_pool1 = max_pool_3x3(h_conv1)#输出为133x66x8
    logging.info(h_pool1)
    ##卷积层conv2
    W_conv2 = weight_variable([3,3,8,6])#第二层卷积：卷积核大小3x3,1个颜色通道，6个卷积核
    b_conv2 = bias_variable([6])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    logging.info(h_conv2)
    h_pool2 = max_pool_3x3(h_conv2)#输出为44x22x6
    logging.info(h_pool2)

    ##全连接层
    h_pool2_flat = tf.reshape(h_pool2,[-1,44*22*6])
    h_fc1 = tf.nn.dropout(h_pool2_flat,dropout)
    h_fc2 = tf.layers.dense(h_fc1, 32,activation=tf.nn.relu)
    h_fc3 = tf.layers.dense(h_fc2, 16,activation=tf.nn.relu)
    
  with tf.variable_scope("concatenate_layer"):
    #词单层bi-lstm和attention拼接
    s1_last = tf.concat([attention1, lstm1_bi_output],1) 
    s2_last = tf.concat([attention2, lstm2_bi_output],1) 
    s3_last = tf.concat([attention3, lstm3_bi_output],1) 

  with tf.variable_scope("Similarity_calculation_layer"):
    def cosine_dist(input1,input2):
        pooled_len_1 = tf.sqrt(tf.reduce_sum(input1 * input1, 1))
        pooled_len_2 = tf.sqrt(tf.reduce_sum(input2 * input2, 1))
        pooled_mul_12 = tf.reduce_sum(input1 * input2, 1)
        score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 +1e-8, name="scores")
        return score

    def manhattan_dist(input1,input2):
        score = tf.exp(-tf.reduce_sum(tf.abs(input1-input2), 1))
        return score
    def multiply(input1,input2):
        score = tf.multiply(input1, input2)  # 矩阵点乘（内积）
        #tf.matmul(matrix3, matrix2)  # 矩阵相乘
        return score
    def subtract(input1,input2):
        score = tf.abs(input1-input2)
        return score
    def maximum(input1,input2):
        s1 = multiply(input1,input1)
        s2 = multiply(input2,input2)
        score = tf.maximum(s1,s2)
        return score

    #词相似度
    cos1 = cosine_dist(s1_last,s2_last)
    cos2 = cosine_dist(s1_last,s3_last)

    man1 = manhattan_dist(s1_last,s2_last)
    man2 = manhattan_dist(s1_last,s3_last)

    mul1 = multiply(s1_last,s2_last)
    mul2 = multiply(s1_last,s3_last)

    sub1 = subtract(s1_last,s2_last)
    sub2 = subtract(s1_last,s3_last)

    maxium1 = maximum(s1_last,s2_last)
    maxium2 = maximum(s1_last,s3_last)

    sub_1 = subtract(lstm1_2_output,lstm2_2_output)
    sub_2 = subtract(lstm1_2_output,lstm3_2_output)

  with tf.variable_scope("dense_layer"):
    last_list_layer = tf.concat([mul1,mul2, sub1,sub2, sub_1,sub_2, maxium1, maxium2],1) 
    #last_list_layer = tf.concat([mul, sub, sub1, maxium ],1) 
    last_drop = tf.nn.dropout(last_list_layer,dropout)
    dense_layer1 = tf.layers.dense(last_drop, 16,activation=tf.nn.relu)
    dense_layer2 = tf.layers.dense(last_drop, 24,activation=tf.nn.sigmoid)
    output = tf.concat([dense_layer1, dense_layer2, tf.expand_dims(cos1,-1), tf.expand_dims(cos2,-1), \
      tf.expand_dims(man1,-1), tf.expand_dims(man2,-1),h_fc3],1)
    #output = tf.concat([dense_layer1, dense_layer2, tf.expand_dims(cos,-1), tf.expand_dims(man,-1)],1)
    dropout_layer = tf.nn.dropout(output, dropout,name='dropout')
    logging.info(dropout_layer)
    fc1 = tf.nn.relu(tf.contrib.layers.linear(dropout_layer, 20))
    logging.info(fc1)
  with tf.variable_scope("classification"):
    logits = tf.layers.dense(fc1, class_nums,activation=None)
    logging.info(logits)
  #计算损失
  with tf.variable_scope("loss"):
    logits_softmax = tf.nn.softmax(logits)
    #losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels)
    #loss = tf.reduce_mean(losses)
    loss = (-weight[0] * tf.reduce_mean(labels[:, 0] * tf.log(logits_softmax[:,0]+1e-10))
            -weight[1] * tf.reduce_mean(labels[:, 1] * tf.log(logits_softmax[:,1]+1e-10))
            -weight[2] * tf.reduce_mean(labels[:, 2] * tf.log(logits_softmax[:,2]+1e-10))
            -weight[3] * tf.reduce_mean(labels[:, 3] * tf.log(logits_softmax[:,3]+1e-10))
            -weight[4] * tf.reduce_mean(labels[:, 4] * tf.log(logits_softmax[:,4]+1e-10))
            )
  #选择优化器
  with tf.variable_scope("train_step"):
    global_add = tf.Variable(0, name="global_step", trainable=False)
    #global_add = global_step.assign_add(1)#用于计数

    #train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    update = tf.train.AdamOptimizer(learning_rate=lr)

    grads_and_vars = update.compute_gradients(loss)
    # 对梯度gradients进行裁剪，保证在[-clip, clip]之间。
    grads_and_vars_clip = [[tf.clip_by_value(g, -clip, clip), v] for g, v in grads_and_vars]
    train_op = update.apply_gradients(grads_and_vars_clip, global_step=global_add)

  #准确率/f1/p/r计算
  with tf.variable_scope("evaluation"):
    true = tf.cast(tf.argmax(labels, axis=-1), tf.float32)#真实序列的值
    pred = tf.cast(tf.argmax(logits, axis=-1), tf.float32)#预测序列的值
    print(pred)
    print(true)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32), name="acc")
    cm = tf.contrib.metrics.confusion_matrix(pred, true, num_classes=class_nums)


def rongcuodu(y_true_list, y_pred_list):
    sum_ = []
    num = 0
    num_dict = {0:0, 1:0, 2:0, 3:0, 4:0}
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        sum_.append(abs(y_true - y_pred))
    for i in sum_:
        num_dict[i] += 1
    for k, v in num_dict.items():
        num_dict[k] = v / len(y_true_list)
    return num_dict

def get_batch(total_sample, batch_size = 128, padding=False, shuffle=True):
  data_order = list(range(total_sample))
  if shuffle:
    np.random.shuffle(data_order)
  if padding:
    if total_sample % batch_size != 0:
      data_order += [data_order[-1]] * (batch_size - total_sample % batch_size)
  for i in range(len(data_order) // batch_size):
    idx = data_order[i * batch_size:(i + 1) * batch_size]
    yield idx
  remain = len(data_order) % batch_size
  if remain != 0:
    idx = data_order[-remain:]
    yield idx



with tf.Session(graph=graph) as sess:
  if isTrain:
    saver = tf.train.Saver(tf.global_variables(),max_to_keep=30)
    if not os.path.isdir(model_path):
      os.mkdir(model_path)

    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt:
        logging.info('Loading model from %s', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        logging.info('Loading model with fresh parameters')
        init = tf.global_variables_initializer()
        sess.run(init)

    max_rho = -float("inf")
    for epoch in range(epoch_num):
      for idx in get_batch(len(train_result_ids),batch_size):
        _,t_loss,t_acc,t_cm,global_nums,t_pred,t_true = sess.run(
            [train_op,loss,acc,cm,global_add,pred,true], {
            X1: train_result_ids[idx],
            X2: train_fw1_ids[idx],
            X3: train_fw2_ids[idx],
            labels : train_labels[idx],
            dropout: 0.9
        })

        if global_nums % 10 == 0:
          logging.info("Train:global_nums:%d,loss:%f,acc:%f,rho:%f" % \
              (global_nums,t_loss,t_acc,pearsonr(t_true, t_pred)[0]))

        if global_nums % 100 == 0:
          val_pred = []
          val_label = []
          for idx in get_batch(len(dev_result_ids),batch_size):
            d_loss,d_pred,d_true = sess.run(
                [loss,pred,true], {
                X1: dev_result_ids[idx],
                X2: dev_fw1_ids[idx],
                X3: dev_fw2_ids[idx],
                labels : dev_labels[idx],
                dropout: 1.0
            })
            val_pred.extend(d_pred)
            val_label.extend(d_true)

          val_rho = pearsonr(val_label, val_pred)[0]
          logging.info("Dev:global_nums:%d,acc:%f,rho:%f" % \
              (global_nums,accuracy_score(val_label, val_pred),val_rho))
          logging.info("容错：")
          logging.info(rongcuodu(val_label, val_pred))
          logging.info("confusion_matrix:")
          logging.info(confusion_matrix(val_label, val_pred))

          #if val_rho > max_rho:
          #  max_rho = val_rho
          path = saver.save(sess, model_path+'/model.ckpt',global_step=global_nums)
          logging.info("save model:%s" % path)
  else:
    graph = tf.Graph()
    sess = tf.Session(graph=graph)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=model_path)
    selected_ckpt = ckpt.all_model_checkpoint_paths[-3]

    saver = tf.train.import_meta_graph(selected_ckpt+'.meta', graph=graph)
    
    saver.restore(sess,selected_ckpt)

    X1 = graph.get_operation_by_name('placeholder/X1').outputs[0]
    X2 = graph.get_operation_by_name('placeholder/X2').outputs[0]
    X3 = graph.get_operation_by_name('placeholder/X3').outputs[0]
    labels = graph.get_operation_by_name('placeholder/Y').outputs[0]
    dropout = graph.get_operation_by_name('placeholder/dropout').outputs[0]
    true = graph.get_operation_by_name('evaluation/Cast').outputs[0]
    pred = graph.get_operation_by_name('evaluation/Cast_1').outputs[0]

    test_pred = []
    test_true = []
    for idx in get_batch(len(test_result_ids),batch_size):
      d_pred,d_true = sess.run(
          [pred,true], {
          X1: test_result_ids[idx],
          X2: test_fw1_ids[idx],
          X3: test_fw2_ids[idx],
          labels : test_labels[idx],
          dropout: 1.0
      })
      test_pred.extend(d_pred)
      test_true.extend(d_true)

    logging.info(len(test_pred))
    logging.info(len(test_true))
    #bad case analyzes
    #logging.info(test_y)
    #logging.info(label_y)
    logging.info("Test:acc:%f,rho:%f" % (accuracy_score(test_true,test_pred),pearsonr(test_true,test_pred)[0]))

    logging.info("容错：")
    logging.info(rongcuodu(test_true,test_pred))
    logging.info("confusion_matrix:")
    cm = confusion_matrix(test_true,test_pred)
    logging.info(cm)
