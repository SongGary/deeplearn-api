import tensorflow as tf
#from tensorflow.contrib.rnn.python.ops.core_rnn_cell import LSTMCell
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import static_rnn
#from tensorflow.contrib.rnn.python.ops.core_rnn import static_rnn
from tensorflow.contrib.legacy_seq2seq import attention_decoder
from tensorflow.contrib.legacy_seq2seq import sequence_loss_by_example, sequence_loss
import numpy as np

import os

def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=False):
    """
    这里需要自己写一个类似函数　因为我们不需要更新解码部分的embedding权值
    :param embedding: 词向量表
    :param output_projection: 输出层的 W B
    :param update_embedding: 是否计算权值更新embedding 一般是False
    :return:
    """
    def loop_function(prev, _):
        if output_projection is not None:
            # Wx + b
            prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
        # max id
        # look up id's vector
        prev_symbol = tf.argmax(prev, 1)
        emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = tf.stop_gradient(emb_prev)
        return emb_prev

    return loop_function

class seq2seqModel(object):
    """
    seq2seq attention 模型 用于文本摘要生成
    目前流传在网络上面的代码有一系列的问题 encode 和 decode 部分的embedding不是同一个
    原因是这部分代码主要是应用是翻译模型 这一部分的代码是目标和实际不一样 因此这里需要注意
    """                
    def __init__(self, vocab_size, buckets, size, num_layers, batch_size, num_softmax_samples, do_decode, num_gpus=2, train_and_test=False):
        """
        :param source_vocab_size:  原始词词数目
        :param target_vocab_size:  目标词词数目
        :param buckets:  桶
        :param size:  cell的神经元数量
        :param num_layers:  神经网络层数
        :param batch_size:
        :param do_decode:  训练还是测试 影响seq2seq的解码过程
        :param num_gpus:  gpu的数量
        :param 训练和预测一起进行
        """
        self._cur_gpu = 0  # 此参数用于自动选择gpu和cpu
        self._num_gpus = num_gpus  # gpu的数量
        self.sess = None  # tf的session 若为None则后面需要创建一个新的
        self.buckets = buckets
        self.global_step = tf.Variable(0, trainable=False)  # 一个tensor 用于记录训练集训练的次数
        
        encoder_inputs = [] #　encoder inputs
        decoder_inputs = []
        target_inputs = []
        loss_weight_inputs =[]

        # 所有的编码输入标识符号
        for i in range(buckets[-1][0]):
            encoder_inputs.append(tf.placeholder(tf.int32, shape=[batch_size], name="encoder{}".format(i)))
        squence_length = tf.placeholder(tf.int32, [batch_size], name='squence_length')
        self.squence_length = squence_length
        # 所有的解码输出标识符号
        for i in range(buckets[-1][1]):
            decoder_inputs.append(tf.placeholder(tf.int32, shape=[batch_size], name="decoder{}".format(i)))
            target_inputs.append(tf.placeholder(tf.int64, shape=[batch_size], name="target{}".format(i)))
            loss_weight_inputs.append(tf.placeholder(tf.float32, shape=[batch_size], name="loss_weight{}".format(i)))
        encoder_inputs_buckets = {}
        decoder_inputs_buckets = {}
        target_inputs_buckets = {}
        loss_weight_inputs_buckets = {}
        # bucket部分的 encoder decoder target
        # 解码和编码部分的bucket
        for bucket_id, bucket in enumerate(buckets):
            encoder_inputs_buckets[bucket_id] = encoder_inputs[0:bucket[0]]
            decoder_inputs_buckets[bucket_id] = decoder_inputs[0:bucket[1]]
            target_inputs_buckets[bucket_id] = target_inputs[0:bucket[1]]
            loss_weight_inputs_buckets[bucket_id] = loss_weight_inputs[0:bucket[1]]

        self.encoder_inputs_buckets = encoder_inputs_buckets
        self.decoder_inputs_buckets = decoder_inputs_buckets
        self.target_inputs_buckets = target_inputs_buckets
        self.loss_weight_inputs_buckets = loss_weight_inputs_buckets

        # 所有的编码部分和解码部分的embedding
        with tf.variable_scope('embedding', reuse=True if train_and_test else None), tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [vocab_size, size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=1e-4))
            # every word look up a word vector.
            emb_encoder_inputs = [tf.nn.embedding_lookup(embedding, x) for x in encoder_inputs]
            emb_decoder_inputs = [tf.nn.embedding_lookup(embedding, x) for x in decoder_inputs]
        encoder_embedding_buckets = {}
        decoder_embedding_buckets = {}
        # bucket embedding 部分的 encoder decoder
        for i, bucket in enumerate(buckets):
            encoder_embedding_buckets[i] = emb_encoder_inputs[0:bucket[0]]
            decoder_embedding_buckets[i] = emb_decoder_inputs[0:bucket[1]]
        # 这里需要使用bucket
        encoder_output_buckets = {}
        encoder_state_buckets = {}
        device = self._next_device()
        for bucket_id, bucket in enumerate(buckets):
            encoder_input_embedding = encoder_embedding_buckets[bucket_id]
            for layer_id in range(num_layers):
                with tf.variable_scope("encoder%d" % layer_id, reuse=(True if bucket_id > 0 else None) or (True if train_and_test else None)), tf.device(device):
                    cell = LSTMCell(num_units=size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123), state_is_tuple=True)
                    encoder_input_embedding, state = static_rnn(cell=cell, inputs=encoder_input_embedding, sequence_length=squence_length, dtype=tf.float32)
                output = encoder_input_embedding
                encoder_output_buckets[bucket_id] = output
                encoder_state_buckets[bucket_id] = state
        with tf.variable_scope('output_projection', reuse=True if train_and_test else None):
            w = tf.get_variable('w', [size, vocab_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=1e-4))
            w_t = tf.transpose(w)
            v = tf.get_variable('v', [vocab_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=1e-4))

        loop_function = _extract_argmax_and_embed(embedding, (w, v)) if do_decode else None
        cell = LSTMCell(size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123), state_is_tuple=True)
        decoder_output_buckets = {}
        decoder_state_buckets = {}
        device = self._next_device()
        for bucket_id, bucket in enumerate(buckets):
            with tf.variable_scope("decoder", reuse=(True if bucket_id > 0 else None) or (True if train_and_test else None)), tf.device(device):
                t = tf.concat(values= [tf.reshape(x, [-1, 1, size]) for x in encoder_output_buckets[bucket_id]], axis = 1)
                decoder_output, decoder_state = attention_decoder(decoder_inputs=decoder_embedding_buckets[bucket_id], initial_state=encoder_state_buckets[bucket_id],
                                  attention_states=t, cell=cell, num_heads=1, loop_function=loop_function, initial_state_attention=do_decode)
                decoder_output_buckets[bucket_id] = decoder_output
                decoder_state_buckets[bucket_id] = decoder_state
        model_output_buckets = {}  # 输出的 logits
        model_output_predict_buckets = {}
        model_output_predict_merger_buckets = {}
        model_output_accuracy = {}
        device = self._next_device()
        for bucket_id, bucket in enumerate(buckets):
            model_output = []
            model_output_predict = []
            model_accuracy = []
            with tf.variable_scope("output", reuse=(True if bucket_id > 0 else None) or (True if train_and_test else None)), tf.device(device):
                for j in range(len(decoder_output_buckets[bucket_id])):
                    output = tf.nn.xw_plus_b(decoder_output_buckets[bucket_id][j], w, v)
                    predict = tf.argmax(input=output, axis=1, name="predict_{}_{}".format(bucket_id, j))
                    accuracy_bool = tf.equal(x=target_inputs_buckets[bucket_id][j], y=predict)
                    model_accuracy.append(tf.reduce_mean(tf.cast(x=accuracy_bool, dtype=tf.float32)))
                    model_output.append(output)
                    model_output_predict.append(tf.reshape(tensor=predict, shape=[-1, 1]))
            model_output_buckets[bucket_id] = model_output
            model_output_predict_buckets[bucket_id] = model_output_predict
            model_output_predict_merger_buckets[bucket_id] = tf.concat(values=model_output_predict, axis=1)
            model_output_accuracy[bucket_id] = tf.add_n(inputs=model_accuracy, name="bucket_id_{}".format(bucket_id)) / \
                                               buckets[bucket_id][1]
        self.model_output_buckets = model_output_buckets
        self.model_output_predict_buckets = model_output_predict_buckets
        self.model_output_predict_merger_buckets = model_output_predict_merger_buckets
        self.model_output_accuracy = model_output_accuracy
        def sampled_loss_func(labels, logits):  # tf1.0的规范更加严格
            with tf.device('/cpu:0'):  # Try gpu.
                labels = tf.reshape(labels, [-1, 1])
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(v, tf.float32)
                local_inputs = tf.cast(logits, tf.float32)
                return tf.cast(tf.nn.sampled_softmax_loss(weights = local_w_t,
                                                  biases = local_b,
                                                  labels = labels,
                                                  inputs = local_inputs,
                                                  num_sampled = num_softmax_samples,
                                                  num_classes = vocab_size), tf.float32)
        device = self._next_device()
        loss_buckets = {}
        for bucket_id, bucket in enumerate(buckets):
            with tf.variable_scope('loss', reuse=(True if bucket_id > 0 else None) or (True if train_and_test else None)), tf.device(device):
                if num_softmax_samples != 0 and not do_decode:
                    # 这里的输入部分不相同的原因是前者替换了softmax函数
                    loss = sequence_loss_by_example(logits = decoder_output_buckets[bucket_id],
                                                    targets = target_inputs_buckets[bucket_id],
                                                    weights = loss_weight_inputs_buckets[bucket_id],
                                                    average_across_timesteps = True,
                                                    softmax_loss_function = sampled_loss_func)
                    # loss = sequence_loss(logits=model_output_buckets[bucket_id],
                    #                      targets=target_inputs_buckets[bucket_id],
                    #                      weights=loss_weight_inputs_buckets[bucket_id]
                    #                      )
                else:
                    loss = sequence_loss(logits = model_output_buckets[bucket_id],
                                         targets = target_inputs_buckets[bucket_id],
                                         weights=loss_weight_inputs_buckets[bucket_id]
                                         )
                loss_buckets[bucket_id] = tf.reduce_mean(loss)  # 计算平均loss
        self.loss_buckets = loss_buckets

    def _next_device(self):  # 自动选择gpu 如果不存在gpu　则全部使用cpu
        if self._num_gpus == 0:
            return '/cpu:0'
        dev = '/gpu:%d' % self._cur_gpu
        if self._num_gpus > 1:
            self._cur_gpu = (self._cur_gpu + 1) % (self._num_gpus)
        return dev

    def _get_gpu(self, gpu_id): # 手动选择gpu 建议使用自动档
        if self._num_gpus <= 0 or gpu_id >= self._num_gpus:
            return ''
        return '/gpu:%d' % gpu_id

    def create_sess(self, sess):
        """
        若sess为None 则新创建一个sess
        :param sess: 
        :return: 
        """
        if sess is None:
            config = tf.ConfigProto(allow_soft_placement=True)
            self.sess = tf.Session(config=config)
        else:
            self.sess = sess

    def fit(self, checkpoint_dir, resultfile, dataset, lr=0.01, min_lr=0.001, max_grad_norm=10.0, epoch=100000):
        """
        
        :param dataset: 
        :param lr: 
        :param min_lr: 
        :param max_grad_norm: 
        :param epoch: 
        :param save: 是否存储模型
        :return: 
        """
        if self.sess is None:
            print("you must create a tensorflow session")
            return
        all_batchs = sum(dataset.buckets_batch_nums.values())
        lr_rate = tf.maximum(min_lr, tf.train.exponential_decay(lr, self.global_step, all_batchs*epoch, min_lr))
        #checkpoint_dir = os.path.abspath("model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        fo = open(resultfile, "w")
        fo.write(checkpoint_prefix);
        fo.close()
        buckets_batch_nums = dataset.buckets_batch_nums
        optimizer = tf.train.AdamOptimizer(lr_rate)
        train_op_buckets = {}
        tvars = tf.trainable_variables()
        for bucket_id, bucket in enumerate(self.buckets):
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(self.loss_buckets[bucket_id], tvars), max_grad_norm)
            train_op_buckets[bucket_id] = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,name='train_step')
        """tensorboard显示一些参数"""
        tf.summary.scalar('learning rate', lr_rate)  # 学习率显示
        merged_buclets = {}
        for bucket_id, bucket in enumerate(self.buckets):
            tf.summary.scalar("loss_{}".format(bucket_id), self.loss_buckets[bucket_id])  # loss显示
            tf.summary.scalar("accuracy_{}".format(bucket_id), self.model_output_accuracy[bucket_id])  # 准确度显示
            merged = tf.summary.merge_all()
            merged_buclets[bucket_id] = merged
            
        process_dir = os.path.abspath(os.path.join(os.path.curdir, 'textsum',"log"))
        writer = tf.summary.FileWriter(process_dir+"/", self.sess.graph)
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=10)
        self.sess.run(tf.initialize_all_variables())
        x_step = 0  # 参数显示的x轴坐标
        for i in range(epoch):
            dataset.shuffle()
            for bucket_id, batch_num in buckets_batch_nums.items():
                for batch_id in range(batch_num):
                    bucket_id, batch_instances_encoder_inputs_ids, batch_instances_decoder_inputs_ids, \
                    batch_instances_targets_ids, batch_instances_weights, batch_instances_length = dataset.next_batch(
                        bucket_id)
                    batch_instances_encoder_inputs_ids = np.transpose(batch_instances_encoder_inputs_ids).tolist()
                    batch_instances_decoder_inputs_ids = np.transpose(batch_instances_decoder_inputs_ids).tolist()
                    batch_instances_targets_ids = np.transpose(batch_instances_targets_ids).tolist()
                    batch_instances_weights = np.transpose(batch_instances_weights).tolist()
                    encoder_inputs_bucket_dict = dict(
                        zip(self.encoder_inputs_buckets[bucket_id], batch_instances_encoder_inputs_ids))
                    decoder_inputs_bucket_dict = dict(
                        zip(self.decoder_inputs_buckets[bucket_id], batch_instances_decoder_inputs_ids))
                    target_inputs_bucket_dict = dict(
                        zip(self.target_inputs_buckets[bucket_id], batch_instances_targets_ids))
                    loss_weight_inputs_bucket_dict = dict(
                        zip(self.loss_weight_inputs_buckets[bucket_id], batch_instances_weights))
                    batch_instances_length_dict = {self.squence_length: batch_instances_length}
                    feed_dict = dict(
                        list(encoder_inputs_bucket_dict.items()) + list(decoder_inputs_bucket_dict.items()) + list(
                            target_inputs_bucket_dict.items())
                        + list(loss_weight_inputs_bucket_dict.items()) + list(batch_instances_length_dict.items()))
                    info, _, loss, result = self.sess.run(
                        [merged_buclets[bucket_id], train_op_buckets[bucket_id], self.loss_buckets[bucket_id],
                         self.model_output_predict_merger_buckets[bucket_id]], feed_dict=feed_dict)
                    writer.add_summary(info, x_step)
                    x_step = x_step + 1  # x坐标增加
                    temp = map(lambda ids: ids[0], batch_instances_encoder_inputs_ids)
                    try:
                        print(i, batch_id, "【训练】输入:{}".format(
                            ", ".join(list(map(lambda x: dataset.all_words[x], temp))).replace("_PAD", "")))
                        print(i, batch_id, "【训练】输出:[ {} ] {}".format(round(loss / 1.0, 5), ", ".join(
                            list(map(lambda x: dataset.all_words[x], result[0]))).replace("_PAD", "").split("_EOS")[0]))
                    except:
                        print(i, batch_id, "【训练】输入:err")
        checkpoint_path = saver.save(sess=self.sess, save_path=checkpoint_prefix, global_step=i)
        print("【训练】结果：", checkpoint_path)

    def predict(self, bucket_id, encoder_length, encoder_inputs_ids, decoder_inputs_ids):
        """
        根据输入的结果预测
        :return: 
        """
        if self.sess is None:
            print("you must create a tensorflow session")
            return
        # 将数据的[batch_size, length] => list([batch_size],[batch_size],..)
        encoder_inputs_ids = np.array(encoder_inputs_ids)
        decoder_inputs_ids = np.array(decoder_inputs_ids)
        encoder_inputs_ids = np.transpose(encoder_inputs_ids)
        decoder_inputs_ids = np.transpose(decoder_inputs_ids)
        encoder_inputs_bucket_dict = dict(zip(self.encoder_inputs_buckets[bucket_id], encoder_inputs_ids))
        decoder_inputs_bucket_dict = {self.decoder_inputs_buckets[bucket_id][0]: decoder_inputs_ids[0]}
        length_dict = {self.squence_length: encoder_length}
        feed_dict = dict(list(encoder_inputs_bucket_dict.items()) + list(decoder_inputs_bucket_dict.items()) + list(length_dict.items()))
        result = self.sess.run([self.model_output_predict_merger_buckets[bucket_id]], feed_dict=feed_dict)
        return result

    def fit_predict_process(self, lr=0.01, min_lr=0.001, max_grad_norm=10.0, epoch=100000, all_batchs=10):
        """
        一边训练一边测试的准备工作
        :return: 
        """
        lr_rate = tf.maximum(min_lr, tf.train.exponential_decay(lr, self.global_step, all_batchs * epoch, min_lr))
        optimizer = tf.train.AdamOptimizer(lr_rate)
        train_op_buckets = {}
        tvars = tf.trainable_variables()
        for bucket_id, bucket in enumerate(self.buckets):
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(self.loss_buckets[bucket_id], tvars),
                                                        max_grad_norm)
            train_op_buckets[bucket_id] = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                    name='train_step')
        """tensorboard显示一些参数"""
        tf.summary.scalar('learning rate', lr_rate)  # 学习率显示
        merged_buclets = {}
        for bucket_id, bucket in enumerate(self.buckets):
            tf.summary.scalar("loss_{}".format(bucket_id), self.loss_buckets[bucket_id])  # loss显示
            tf.summary.scalar("accuracy_{}".format(bucket_id), self.model_output_accuracy[bucket_id])  # 准确度显示
            merged = tf.summary.merge_all()
            merged_buclets[bucket_id] = merged
        writer = tf.summary.FileWriter("log/", self.sess.graph)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        self.sess.run(tf.global_variables_initializer())
        self.train_op_buckets = train_op_buckets
        self.merged_buclets = merged_buclets
        self.writer = writer
        self.saver = saver

    def fit_predict(self, dataset, epoch, save=False):
        """
        调用此方法必须首先调用fit_predict_process(...)
        :param dataset: 
        :param epoch: 
        :return: 
        """
        buckets_batch_nums = dataset.buckets_batch_nums
        checkpoint_dir = os.path.abspath("model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        x_step = 0  # 参数显示的x轴坐标
        for i in range(epoch):
            dataset.shuffle()
            for bucket_id, batch_num in buckets_batch_nums.items():
                for batch_id in range(batch_num):
                    bucket_id, batch_instances_encoder_inputs_ids, batch_instances_decoder_inputs_ids, \
                    batch_instances_targets_ids, batch_instances_weights, batch_instances_length = dataset.next_batch(
                        bucket_id)
                    batch_instances_encoder_inputs_ids = np.transpose(batch_instances_encoder_inputs_ids).tolist()
                    batch_instances_decoder_inputs_ids = np.transpose(batch_instances_decoder_inputs_ids).tolist()
                    batch_instances_targets_ids = np.transpose(batch_instances_targets_ids).tolist()
                    batch_instances_weights = np.transpose(batch_instances_weights).tolist()
                    encoder_inputs_bucket_dict = dict(
                        zip(self.encoder_inputs_buckets[bucket_id], batch_instances_encoder_inputs_ids))
                    decoder_inputs_bucket_dict = dict(
                        zip(self.decoder_inputs_buckets[bucket_id], batch_instances_decoder_inputs_ids))
                    target_inputs_bucket_dict = dict(
                        zip(self.target_inputs_buckets[bucket_id], batch_instances_targets_ids))
                    loss_weight_inputs_bucket_dict = dict(
                        zip(self.loss_weight_inputs_buckets[bucket_id], batch_instances_weights))
                    batch_instances_length_dict = {self.squence_length: batch_instances_length}
                    feed_dict = dict(
                        list(encoder_inputs_bucket_dict.items()) + list(decoder_inputs_bucket_dict.items()) + list(
                            target_inputs_bucket_dict.items())
                        + list(loss_weight_inputs_bucket_dict.items()) + list(batch_instances_length_dict.items()))
                    info, _, loss, result = self.sess.run(
                        [self.merged_buclets[bucket_id], self.train_op_buckets[bucket_id], self.loss_buckets[bucket_id],
                         self.model_output_predict_merger_buckets[bucket_id]], feed_dict=feed_dict)
                    self.writer.add_summary(info, x_step)
                    x_step = x_step + 1  # x坐标增加
                    temp = map(lambda ids: ids[0], batch_instances_encoder_inputs_ids)
                    try:
                        print(i, batch_id, "【训练】输入:{}".format(
                            ", ".join(list(map(lambda x: dataset.all_words[x], temp))).replace("_PAD", "")))
                        print(i, batch_id, "【训练】输出:[ {} ] {}".format(round(loss / 1.0, 5), ", ".join(
                            list(map(lambda x: dataset.all_words[x], result[0]))).replace("_PAD", "").split("_EOS")[0]))
                    except:
                        print(i, batch_id, "【训练】输入:err")
        if save:
            checkpoint_path = self.saver.save(sess=self.sess, save_path=checkpoint_prefix, global_step=i)
            print("【训练】结果：", checkpoint_path)



if __name__ == "__main__":
    pass
    # vocab_size = 10
    # buckets = buckets = [(30, 20), (40, 20)]
    # size = 64
    # num_layers = 2
    # batch_size = 1
    # num_softmax_samples = 10
    # do_decode = False
    # model = seq2seqModel(vocab_size, buckets, size, num_layers, batch_size, num_softmax_samples, do_decode)