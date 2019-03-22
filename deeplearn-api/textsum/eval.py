import tensorflow as tf

from seq2seq_attention_model import seq2seqModel
from datasets import DataSet, DataConvert
import json
import argparse
import os

#set argparser
"""
深度学习 seq2seq + attention　模型
"""
def parse_args():
    parser = argparse.ArgumentParser(description="seq2seq + attention")
    
    parser.add_argument("--dataPath")
    parser.add_argument("--modelPath")
    parser.add_argument("--outputPath")
    parser.add_argument("--resultfile")
    arguments = parser.parse_args()
    dataPath = arguments.dataPath
    print("dataPath: ",dataPath)  
    modelPath = arguments.modelPath
    print("modelPath: ",modelPath)  
    outputPath = arguments.outputPath
    print("outputPath: ",outputPath) 
    resultfile = arguments.resultfile
    print("resultfile: ",resultfile)
    
    parser.add_argument('--num_gpus', help='numbers of gpu, defalut=2', default=2, type=int)
    parser.add_argument('--epoch', help='numbers of epoch, default=100 ', default=1000, type=int)
    parser.add_argument('--exists', help='data exists yes or no, defalut=False', default=False, type=bool)
    parser.add_argument('--buckets', help='default=[(30, 20)]', default=[(30, 20)], type=list)
    parser.add_argument('--batch_size', help='default=1', default=6, type=int)
    parser.add_argument('--content', help='content data file', default=dataPath, type=str)
    parser.add_argument('--size', help='lstm unit size default=128', default=128, type=int)
    parser.add_argument('--num_layers', help='encoder layer size', default=2, type=int)
    parser.add_argument('--do_decoder', help='do decoder', default=False, type=bool)
    parser.add_argument('--lr', help='learning rate, default=0.01', default=0.01, type=float)
    parser.add_argument('--min_lr', help='min learning rate, default=0.001', default=0.001, type=float)
    parser.add_argument('--max_grad_norm', help='max grad norm, default=10.0', default=10.0, type=float)
    parser.add_argument('--train_or_test', help='train, test or train_test, default=train', default="test", type=str)
    return parser.parse_args()  # 这里需要特别留意 python2和python3是有一些区别的

def test(dataPath, modelPath, outputPath):
    conf_dir = os.path.abspath(os.path.join(os.path.curdir, 'textsum',"conf"))
    params = json.load(open(conf_dir+"/params.json", "r"))
    buckets = params['buckets']
    vocab_size = params['vocab_size']
    size = params['size']
    num_softmax_samples = params['num_softmax_samples']
    num_layers = params['num_layers']
    batch_size = 1
    do_decoder = True  # 预测部分
    num_gpus = 0
    #num_gpus = params['num_gpus']
    convert = DataConvert(buckets)
    fo = open(dataPath, "r+")
    test_str = fo.read()
    #test_str = '据悉，朝鲜最高领袖金正恩今日访华'
    bucket_id, encoder_length, encoder_inputs_ids, decoder_inputs_ids= convert.convert(test_str)
    model = seq2seqModel(vocab_size, buckets, size, num_layers, batch_size, num_softmax_samples, do_decoder, num_gpus)
    #checkpoint_dir = os.path.abspath("model")
    checkpoint_file = tf.train.latest_checkpoint(modelPath)  # 这里取出的是最后一个model文件
    print("checkpoint_file: ",checkpoint_file)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, checkpoint_file)  # 恢复sess模型
    model.create_sess(sess=sess)
    result = model.predict(bucket_id, encoder_length, encoder_inputs_ids, decoder_inputs_ids)
    print("输入数据:{}".format(test_str))
    print("输出数据:{}".format(", ".join(list(map(lambda x: convert.all_words[x], result[0][0])))))
    with open(outputPath, 'w') as f:
        f.write("摘要:{}".format(", ".join(list(map(lambda x: convert.all_words[x], result[0][0])))))
    f.close()


def train_test(buckets, batch_size, content, title, size, num_layers, do_decoder, num_gpus, exists, epoch, lr, min_lr, max_grad_norm):
    """
    支持一边训练一边测试结果 但是testmodel必须事先调用fit_predict_process
    :param buckets: 
    :param batch_size: 
    :param content: 
    :param title: 
    :param size: 
    :param num_layers: 
    :param do_decoder: 
    :param num_gpus: 
    :param exists: 
    :param epoch: 
    :param lr: 
    :param min_lr: 
    :param max_grad_norm: 
    :return: 
    """
    dataset = DataSet(content, title, batch_size, exists, buckets)
    vocab_size = dataset.vocab_size
    num_softmax_samples = int(dataset.vocab_size / 3)  # python3需要转换成int
    convert = DataConvert(buckets)
    test_str = '唱吧是一个手机KTV唱歌娱乐应用。北京最淘科技有限公司旗下产品。'
    bucket_id, encoder_length, encoder_inputs_ids, decoder_inputs_ids = convert.convert(test_str)
    do_decoder = False
    # 将参数存储成json格式文件
    params = {"buckets": buckets, "batch_size": batch_size, "size": size, "num_layers": num_layers,
              "do_decoder": do_decoder,
              "num_gpus": num_gpus, "exists": exists, "epoch": epoch, "vocab_size": vocab_size,
              "num_softmax_samples": num_softmax_samples}
    if not os.path.exists("conf"):
        os.mkdir("conf")
    json.dump(params, open("conf/params.json", "w"))
    train_model = seq2seqModel(vocab_size, buckets, size, num_layers, batch_size, num_softmax_samples, do_decoder, num_gpus)
    train_model.create_sess(sess=None)
    all_batchs = sum(dataset.buckets_batch_nums.values())
    train_model.fit_predict_process(lr, min_lr, max_grad_norm, epoch, all_batchs)  # 必要的预处理工作　(主要部分是参数初始化)
    sess = train_model.sess  # 获取第一个model的sess文件作为全局sess
    do_decoder = True
    # 注意　这里的train_and_test必须设置为True batch_size=1
    test_model = seq2seqModel(vocab_size, buckets, size, num_layers, 1, num_softmax_samples, do_decoder, num_gpus, train_and_test=True)
    test_model.create_sess(sess=sess)
    save = False
    for i in range(epoch):
        # 训练一次
        print("start train epoch:{}".format(i))
        if i == epoch-1:  # 最后一次训练需要保存模型文件
            train_model.fit_predict(dataset, 1, True)
        else:
            train_model.fit_predict(dataset, 1)
        print("start predict epoch:{}".format(i))
        result = test_model.predict(bucket_id, encoder_length, encoder_inputs_ids, decoder_inputs_ids)
        print("输入数据:{}".format(test_str))
        print("输出数据:{}".format(", ".join(list(map(lambda x: convert.all_words[x], result[0][0])))))

def main(args):
    num_gpus = args.num_gpus
    exists = args.exists
    buckets = args.buckets
    batch_size = args.batch_size
    content = args.content
    size = args.size
    num_layers = args.num_layers
    do_decoder = args.do_decoder
    epoch = args.epoch
    trian_or_test = args.train_or_test
    lr = args.lr
    min_lr = args.min_lr
    max_grad_norm = args.max_grad_norm
    dataPath = args.dataPath
    modelPath = args.modelPath
    outputPath = args.outputPath
    if trian_or_test == 'test':
        test(dataPath, modelPath, outputPath)
    elif trian_or_test == 'train_test':
        train_test(buckets, batch_size, content, size, num_layers, do_decoder, num_gpus, exists, epoch, lr,min_lr, max_grad_norm)
    else:
        print("python main.py --help")
if __name__ == '__main__':
    args = parse_args()
    main(args)