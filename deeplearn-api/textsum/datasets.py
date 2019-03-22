import re
import os
import pickle
import jieba
from process import *
import numpy as np
import collections

# 特殊字符标记
_PAD = "_PAD"  # 填充符号
_UNK = "_UNK"  # 未登录词
_GO = "_GO"
_EOS = "_EOS"
_START_VOCAB = [_PAD, _UNK, _GO, _EOS]

stop_symbol = '()，．。！,ゅ\'～~`·？-=:、‘；[]{}*&^%$#@!~+_=-！@#￥%……&*（）——+}{【】||？。><！，;.1234567890：“”"》《+?/%）（@ \n \t \u3000'

out_dir = os.path.abspath(os.path.join(os.path.curdir, 'textsum',"stopwords.txt"))
f_stop = open(out_dir,encoding= 'utf-8')  
try:  
    f_stop_text = f_stop.read( )
finally:  
    f_stop.close( ) 
f_stop_seg_list=f_stop_text.split('\n')

process_dir = os.path.abspath(os.path.join(os.path.curdir, 'textsum',"process_data"))

class DataSet(object):

    def __init__(self, train_data, test_data, batch_size, data_exists, buckets, test_nums=1):
        """
        :param train_data: 训练集
        :param test_data:  测试集
        :param batch_size:
        :param data_exists: 数据是否已经存在
        :param buckets: 桶 解码输出部分 必须保证是由小到大排序
        :param test_nums 测试样本
        """
        self.buckets = buckets
        content_word_lengths = list(map(lambda encoder_decoder_length: encoder_decoder_length[0], buckets))
        self.content_word_lengths = content_word_lengths
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.test_nums = test_nums
        if not data_exists:
            print("process...")
            content_max_length = buckets[-1][0]
            title_max_length = buckets[-1][1]
            self.process(content_max_length = content_max_length, title_max_length = title_max_length)
        else:
            print("load...")
            self.load()
        self.bucket_process()  # 根据数据生成bucket

    def process(self, content_max_length=200, title_max_length=40):
        """
        这里对数据进行了一些限制　content 长度小于200 title　长度小于40
        :param content_max_length 正文最大长度
        :param title_max_length  标题最大长度
        :return:
        """
        self.content_max_length = content_max_length
        self.title_max_length = title_max_length

        with open(self.test_data, "r", encoding="utf-8") as f:
            titles = f.readlines()
        with open(self.train_data, "r", encoding="utf-8") as f:
            contents = f.readlines()
        titles = list(map(lambda title: title.strip(), titles))
        titles = list(map(lambda title: replace_t_num(title), titles))
        titles = list(map(lambda title: replace_data(title), titles))
        titles = list(map(lambda title: replace_num(title), titles))
        contents = list(map(lambda content: content.strip(), contents))
        contents = list(map(lambda content: replace_t_num(content), contents))
        contents = list(map(lambda content: replace_data(content), contents))
        contents = list(map(lambda content: replace_num(content), contents))
        all_words = set()
        instances_titles = []
        instances_contents = []
        for title, content in zip(titles, contents):
#             if content.__contains__("中国雅虎侠客平台仅提供信息存储空间服务，其内容均由上网用户提供。"):
#                 continue
            datat = jieba.cut(title, cut_all=False)
            title_words = []
            buffer = "/".join(datat)
            da = buffer.split("/")
            for symbol in da:
                if symbol not in f_stop_seg_list and symbol not in stop_symbol:
                    title_words.append(symbol)
            
            title_words = list(map(lambda word: word, title_words))
            
            datac = jieba.cut(content, cut_all=False)
            content_words = []
            bufferc = "/".join(datac)
            dac = bufferc.split("/")
            for symbol in dac:
                if symbol not in f_stop_seg_list and symbol not in stop_symbol:
                    content_words.append(symbol)
            
            content_words = list(map(lambda word: word, content_words))
            if len(title_words) > title_max_length or len(content_words) > content_max_length:  # 针对正文和标题的长度进行限制
                continue
            instances_titles.append(title_words)
            instances_contents.append(content_words)
            for word in title_words + content_words:
                if word not in all_words:
                    all_words.add(word)
        
        
        if not os.path.exists(process_dir):
            os.mkdir(process_dir)
        all_words = list(all_words) + _START_VOCAB
        vocab_size = len(all_words)
        words_id = dict(zip(all_words, range(vocab_size)))
        pickle.dump(all_words, open(process_dir+"/all_words", "wb"))
        pickle.dump(words_id, open(process_dir+"/words_id", "wb"))
        pickle.dump(instances_titles, open(process_dir+"/instances_titles", "wb"))
        pickle.dump(instances_contents, open(process_dir+"/instances_contents", "wb"))
        test_titles = instances_titles[-self.test_nums:]  # 取出最后几行当作测试集
        test_contents = instances_contents[-self.test_nums:]  # 取出最后几行当作测试集
        instances_titles = instances_titles[:-self.test_nums]
        instances_contents = instances_contents[:-self.test_nums]
        self.test_titles = test_titles
        self.test_contents = test_contents
        self.all_words = all_words
        self.vocab_size = vocab_size
        self.words_id = words_id
        self.instances_titles = instances_titles
        self.instances_contents = instances_contents
        self.num_instances = len(instances_contents)

    def load(self):
        all_words = pickle.load(open(process_dir+"/all_words", "rb"))
        words_id = pickle.load(open(process_dir+"/words_id", "rb"))
        instances_titles = pickle.load(open(process_dir+"/instances_titles", "rb"))
        instances_contents = pickle.load(open(process_dir+"/instances_contents", "rb"))
        vocab_size = len(all_words)
        test_titles = instances_titles[-self.test_nums:]  # 取出最后几行当作测试集
        test_contents = instances_contents[-self.test_nums:]  # 取出最后几行当作测试集
        instances_titles = instances_titles[:-self.test_nums]
        instances_contents = instances_contents[:-self.test_nums]
        self.test_titles = test_titles
        self.test_contents = test_contents
        self.all_words = all_words
        self.vocab_size = vocab_size
        self.words_id = words_id
        self.instances_titles = instances_titles
        self.instances_contents = instances_contents
        self.num_instances = len(instances_contents)


    def bucket_process(self):
        """
        生成必要的bucket 这里针对数据进行了一些筛选  选择一部分范围内的数据 拍脑袋决定的 别问我是为啥 这里是随机编写的一种
        :return:
        """
        # 针对数据进行必要的过滤工作　这里的encode部分限制在0~200之间
        print ("instance size:{}".format(len(self.instances_contents)))
        print ("vocab_size:{}".format(self.vocab_size))
        contents_length = map(lambda content: len(content), self.instances_contents)
        print ("content length details:{}".format(collections.Counter(contents_length)))
        titles_length = map(lambda title: len(title), self.instances_titles)
        print ("title length details:{}".format(collections.Counter(titles_length)))
        buckets_instances_titles = {}
        buckets_test_titles = {}
        buckets_instances_contents = {}
        buckets_test_contents = {}
        for i, (_, _) in enumerate(self.buckets):
            buckets_instances_titles[i] = []
            buckets_instances_contents[i] = []
            buckets_test_titles[i] = []
            buckets_test_contents[i] = []
        for content, title in zip(self.test_titles, self.test_contents):
            for id, length in enumerate(self.content_word_lengths):
                if len(content) < length:
                    buckets_test_titles[id].append(title)
                    buckets_test_contents[id].append(content)
                    break  # 跳出循环
        for content, title in zip(self.instances_contents, self.instances_titles):
            for id, length in enumerate(self.content_word_lengths):
                if len(content) < length:
                    buckets_instances_titles[id].append(title)
                    buckets_instances_contents[id].append(content)
                    break  # 跳出循环
        self.buckets_test_titles = buckets_test_titles
        self.buckets_test_contents = buckets_test_contents
        self.buckets_instances_titles = buckets_instances_titles
        self.buckets_instances_contents = buckets_instances_contents
        buckets_instances_numbers = {}
        for id, sub_instances_contents in buckets_instances_contents.items():
            buckets_instances_numbers[id] = len(sub_instances_contents)
        print ("buckets_instances_numbers:{}".format(buckets_instances_numbers))
        self.buckets_instances_numbers = buckets_instances_numbers
        buckets_instances_indexs = {}
        buckets_points = {}
        buckets_batch_nums = {}
        for id, length in buckets_instances_numbers.items():
            buckets_instances_indexs[id] = np.arange(length)
            buckets_points[id] = 0
            buckets_batch_nums[id] = int(length / self.batch_size)
        self.buckets_instances_indexs = buckets_instances_indexs
        self.buckets_points = buckets_points
        self.buckets_batch_nums = buckets_batch_nums
        print ("bucket_id: instance numbers -> {}".format(map(lambda id_index: np.shape(id_index[1]), buckets_instances_indexs.items())))
        print ("buckets_points:{}".format(buckets_points))
        print ("buckets_instances_indexs:{}".format(buckets_instances_indexs[0]))
        print ("buckets_batch_nums:{}".format(buckets_batch_nums))

    def shuffle(self):
        for id, length in self.buckets_instances_numbers.items():
            self.buckets_points[id] = 0
            np.random.shuffle(self.buckets_instances_indexs[id])
        # print("shuffle success!!")

    def next_batch(self, bucket_id):
        """
        这里的batch_size 与 bucket不会出现越界的情况　可以放心使用
        """
        start = self.buckets_points[bucket_id]
        self.buckets_points[bucket_id] = start + self.batch_size
        end = self.buckets_points[bucket_id]
        # bucket的输入和输出长度
        encoder_length, decoder_length = self.buckets[bucket_id]
        # 对应index位置的数据
        batch_instances_titles = list(map(lambda index: self.buckets_instances_titles[bucket_id][index], self.buckets_instances_indexs[bucket_id][start: end]))
        # 添加GO起始符号
        batch_instances_decoder_inputs = list(map(lambda instances_titles: [_GO] + instances_titles, batch_instances_titles))
        # 数据填充PAD
        batch_instances_decoder_inputs = list(map(lambda instances_titles: instances_titles + (decoder_length - len(instances_titles))*[_PAD], batch_instances_decoder_inputs))
        batch_instances_decoder_inputs_ids = map(lambda batch_instances_decoder_inputs: map(lambda word: self.words_id[word] if word in self.words_id else self.words_id[_UNK],batch_instances_decoder_inputs), batch_instances_decoder_inputs)
        # 将map转换成list
        batch_instances_decoder_inputs_ids = [list(instances_decoder_inputs_ids) for instances_decoder_inputs_ids in list(batch_instances_decoder_inputs_ids)]
        # 添加EOS结束符号
        batch_instances_targets = list(map(lambda instances_titles: instances_titles + [_EOS], batch_instances_titles))
        # 计算loss weights
        batch_instances_weights = list(map(lambda instances_targets: len(instances_targets) * [1.0] + (decoder_length - len(instances_targets)) * [0.0], batch_instances_targets))
        # print batch_instances_weights[0]
        # print map(lambda batch_instances:len(batch_instances), batch_instances_weights)
        # 数据填充
        batch_instances_targets = list(map(lambda instances_targets: instances_targets + (decoder_length-len(instances_targets))*[_PAD], batch_instances_targets))
        # word -> id
        batch_instances_targets_ids = map(lambda instances_targets: map(lambda word: self.words_id[word] if word in self.words_id else self.words_id[_UNK], instances_targets),batch_instances_targets)
        batch_instances_targets_ids = [list(instances_targets_ids) for instances_targets_ids in batch_instances_targets_ids]

        # 对应index位置的数据
        batch_instances_contents = list(map(lambda index: self.buckets_instances_contents[bucket_id][index], self.buckets_instances_indexs[bucket_id][start: end]))
        """这里的长度有一些问题。。需要注意"""
        # batch_instances_length = map(lambda instances_contents: len(instances_contents), batch_instances_contents)
        batch_instances_length = list(map(lambda instances_contents: encoder_length, batch_instances_contents))
        # 数据填充PAD
        batch_instances_contents = list(map(lambda instances_contents: (encoder_length-len(instances_contents))*[_PAD] + instances_contents, batch_instances_contents))
        # word -> id
        batch_instances_encoder_inputs_ids = map(lambda instances_contents: map(lambda word: self.words_id[word] if word in self.words_id else self.words_id[_UNK], instances_contents),batch_instances_contents)
        batch_instances_encoder_inputs_ids = [ list(instances_encoder_inputs_ids) for instances_encoder_inputs_ids in batch_instances_encoder_inputs_ids]

        assert len(batch_instances_decoder_inputs_ids) == len(batch_instances_targets_ids) == len(batch_instances_weights) == len(batch_instances_encoder_inputs_ids)
        """
        bucket_id
        batch_instances_encoder_inputs_ids  [batch_size, encoder_inputs_length]
        batch_instances_decoder_inputs_ids  [batch_size, decoder_inputs_length]
        batch_instances_targets_ids [batch_size, decoder_targets_length]  # decoder_targets_length == decoder_inputs_length
        batch_instances_weights [batch_size, encoder_inputs_length]
        batch_instances_length [batch_size]
        """
        return bucket_id, batch_instances_encoder_inputs_ids, batch_instances_decoder_inputs_ids, batch_instances_targets_ids, batch_instances_weights, batch_instances_length

    def get_test(self, bucket_id):
        # bucket的输入和输出长度
        encoder_length, decoder_length = self.buckets[bucket_id]
        batch_instances_titles = self.buckets_test_titles[bucket_id]
        # 添加GO起始符号
        batch_instances_decoder_inputs = list(
            map(lambda instances_titles: [_GO] + instances_titles, batch_instances_titles))
        # 数据填充PAD
        batch_instances_decoder_inputs = list(
            map(lambda instances_titles: instances_titles + (decoder_length - len(instances_titles)) * [_PAD],
                batch_instances_decoder_inputs))
        batch_instances_decoder_inputs_ids = map(lambda batch_instances_decoder_inputs: map(
            lambda word: self.words_id[word] if word in self.words_id else self.words_id[_UNK],
            batch_instances_decoder_inputs), batch_instances_decoder_inputs)
        # 将map转换成list
        batch_instances_decoder_inputs_ids = [list(instances_decoder_inputs_ids) for instances_decoder_inputs_ids in
                                              list(batch_instances_decoder_inputs_ids)]
        # 添加EOS结束符号
        batch_instances_targets = list(map(lambda instances_titles: instances_titles + [_EOS], batch_instances_titles))
        # 计算loss weights
        batch_instances_weights = list(map(
            lambda instances_targets: len(instances_targets) * [1.0] + (decoder_length - len(instances_targets)) * [
                0.0], batch_instances_targets))
        # print batch_instances_weights[0]
        # print map(lambda batch_instances:len(batch_instances), batch_instances_weights)
        # 数据填充
        batch_instances_targets = list(
            map(lambda instances_targets: instances_targets + (decoder_length - len(instances_targets)) * [_PAD],
                batch_instances_targets))
        # word -> id
        batch_instances_targets_ids = map(lambda instances_targets: map(
            lambda word: self.words_id[word] if word in self.words_id else self.words_id[_UNK], instances_targets),
                                          batch_instances_targets)
        batch_instances_targets_ids = [list(instances_targets_ids) for instances_targets_ids in
                                       batch_instances_targets_ids]
        # 对应index位置的数据
        batch_instances_contents = self.buckets_test_contents[bucket_id]
        """这里的长度有一些问题。。需要注意"""
        # batch_instances_length = map(lambda instances_contents: len(instances_contents), batch_instances_contents)
        batch_instances_length = list(map(lambda instances_contents: encoder_length, batch_instances_contents))
        # 数据填充PAD
        batch_instances_contents = list(
            map(lambda instances_contents: (encoder_length - len(instances_contents)) * [_PAD] + instances_contents,
                batch_instances_contents))
        # word -> id
        batch_instances_encoder_inputs_ids = map(lambda instances_contents: map(
            lambda word: self.words_id[word] if word in self.words_id else self.words_id[_UNK], instances_contents),
                                                 batch_instances_contents)
        batch_instances_encoder_inputs_ids = [list(instances_encoder_inputs_ids) for instances_encoder_inputs_ids in
                                              batch_instances_encoder_inputs_ids]

        assert len(batch_instances_decoder_inputs_ids) == len(batch_instances_targets_ids) == len(
            batch_instances_weights) == len(batch_instances_encoder_inputs_ids)
        """
        bucket_id
        batch_instances_encoder_inputs_ids  [batch_size, encoder_inputs_length]
        batch_instances_decoder_inputs_ids  [batch_size, decoder_inputs_length]
        batch_instances_targets_ids [batch_size, decoder_targets_length]  # decoder_targets_length == decoder_inputs_length
        batch_instances_weights [batch_size, encoder_inputs_length]
        batch_instances_length [batch_size]
        """
        return bucket_id, batch_instances_encoder_inputs_ids, batch_instances_decoder_inputs_ids, batch_instances_targets_ids, batch_instances_weights, batch_instances_length


class DataConvert(object):
    """
            数据的转换 用于模型的预测阶段
    """
    def __init__(self, buckets):
        """

        :param buckets: 桶
        """
        self.buckets = buckets
        content_word_lengths = list(map(lambda encoder_decoder_length: encoder_decoder_length[0], buckets))
        self.content_word_lengths = content_word_lengths
        content_max_length = buckets[-1][0]
        title_max_length = buckets[-1][1]
        self.content_max_length = content_max_length
        self.title_max_length = title_max_length
        self.load()

    def load(self):
        all_words = pickle.load(open(process_dir+"/all_words", "rb"))
        words_id = pickle.load(open(process_dir+"/words_id", "rb"))
        vocab_size = len(all_words)
        self.all_words = all_words
        self.vocab_size = vocab_size
        self.words_id = words_id

    def convert(self, content):
        content_words = jieba.cut(content)
        content_words = list(map(lambda word: word, content_words))
        try:
            assert (len(content_words) <= self.content_max_length)  # 限定最大长度 不可超过这个门限值
        except:
            print(len(content_words),">",self.content_max_length)
        # 找到对应的bucket
        for id, length in enumerate(self.content_word_lengths):
            if len(content_words) < length:
                break
        encoder_length, decoder_length = self.buckets[id]  # 找出对应的bucket的 编码和解码长度
        decoder_inputs = [_GO] * decoder_length
        content_words = [_PAD]*(encoder_length - len(content_words)) + content_words
        encoder_inputs_ids = list(map(lambda x:self.words_id[x] if x in self.words_id else self.words_id[_UNK], content_words))
        # 解码输入部分 实际应用当中应该只会用到第一个Go
        decoder_inputs_ids = list(map(lambda x:self.words_id[x] if x in self.words_id else self.words_id[_UNK], decoder_inputs))
        encoder_length = encoder_length
        return id, [encoder_length], [encoder_inputs_ids], [decoder_inputs_ids]  # batch_size=1


if __name__ == "__main__":
    pass
    buckets = [(30, 20)]
    content_word_lengths = [30]
    batch_size = 1
    train_data = "data/contents.txt"
    test_data = "data/titles.txt"
    dataset = DataSet(train_data, test_data, batch_size, False, buckets, test_nums=1)
    print(dataset.buckets_instances_indexs)
    dataset.shuffle()
    print(dataset.buckets_instances_indexs)
    # bucket_id, batch_instances_encoder_inputs_ids, batch_instances_decoder_inputs_ids, batch_instances_targets_ids, batch_instances_weights, batch_instances_length = \
    # dataset.get_test(0)
    # print(", ".join(list(map(lambda x:dataset.all_words[x], batch_instances_encoder_inputs_ids[0]))))
    # print(", ".join(list(map(lambda x:dataset.all_words[x], batch_instances_decoder_inputs_ids[0]))))
    # print(", ".join(list(map(lambda x:dataset.all_words[x], batch_instances_targets_ids[0]))))
    # print(batch_instances_weights)
    # print(batch_instances_length)