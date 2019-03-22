# coding=utf-8
"""
数据预处理
"""
import os


def test(file):
    for f in os.listdir(file):
        sub_f = os.path.join(file, f)
        if os.path.isdir(sub_f):
            test(sub_f)
        else:
            print (sub_f)
import re

def get_files(data):
    files = []
    for file in os.listdir(data):
        if file.endswith(".txt"):
            files.append(file)
    return files

_DIGIT_RE = re.compile("[\d]+")  # 用于处理数字符号
_DATA_RE1 = re.compile("[\d〇一二三四五六七八九十]{0,4}年[\d〇一二三四五六七八九十]{0,3}月[\d〇一二三四五六七八九十]{0,3}日")
_DATA_RE2 = re.compile("[\d〇一二三四五六七八九十]{0,4}年[\d〇一二三四五六七八九十]{0,3}月")
_DATA_RE3 = re.compile("[\d〇一二三四五六七八九十]{0,3}月[\d〇一二三四五六七八九十]{0,3}日")

def replace_t_num(str):
    """
    特殊字符的特换
    :param str:
    :return:
    """
    return str.replace('１', '1').replace('２', '2').replace('３', '3').replace('４', '4').replace('５', '5').replace('６', '6').replace('７', '7').replace('８', '8').replace('９', '9').replace('０', '0')

def replace_num(str):
    return re.sub(_DIGIT_RE, "NUM", str)

def replace_data(str):
    str = re.sub(_DATA_RE1, "DATA", str)
    str = re.sub(_DATA_RE2, "DATA", str)
    str = re.sub(_DATA_RE3, "DATA", str)
    return str

# def process(title_path, content_path):
#     with open("data/titles.txt", "r") as f:
#         titles = f.readlines()
#     with open("data/contents.txt", "r") as f:
#         contents = f.readlines()
#     titles = map(lambda title: replace_t_num(title), titles)
#     contents = map(lambda content: replace_t_num(content), contents)


if __name__ == "__main__":
    with open("data/titles.txt", encoding="utf-8") as f:
        titles = f.readlines()
    with open("data/contents.txt", encoding="utf-8") as f:
        contents = f.readlines()
    titles = map(lambda title: replace_t_num(title), titles)
    titles = map(lambda title: replace_data(title), titles)
    titles = map(lambda title: replace_num(title), titles)
    contents = map(lambda content: replace_t_num(content), contents)
    contents = map(lambda content: replace_data(content), contents)
    contents = map(lambda content: replace_num(content), contents)
    print(list(titles))
    print(list(contents))