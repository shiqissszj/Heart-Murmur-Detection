import argparse
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from DataProcessing.find_and_load_patient_files import (
    find_patient_files,
    load_patient_data,
)
from DataProcessing.label_extraction import get_murmur, get_outcome, get_murmurmost, get_murmurlocation
from DataProcessing.XGBoost_features.metadata import get_metadata


def stratified_test_vali_split(
        stratified_features: list,
        data_directory: str,
        out_directory: str,
        test_size: float,
        vali_size: float,
        random_state: int,
):
    # Check if out_directory directory exists, otherwise create it.
    # 检查文件夹是否存在如果没有存在就创建他
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    else:
        shutil.rmtree(out_directory)
        # 使用树结构递归删除文件，文件夹下面的所有文件
        os.makedirs(out_directory)
    # Get metadata
    patient_files = find_patient_files(data_directory)
    # 获取所有文件的文件名信息组成的二维数组，格式为字符串

    num_patient_files = len(patient_files)
    # print(num_patient_files)
    # 在这里输出值为942 所有文件的数组
    murmur_classes = ["Present", "Unknown", "Absent"]
    # 这里定义音频的标签 总共三种，存在，位置，缺席
    murmur2_classes = ["Systolic", "Diastolic"]

    num_murmur2_classes = len(murmur2_classes)

    # 将声音分为 收缩期和伸张期

    murmur3_classes = ["0000", "0010", "0100", "1000"]
    # 分别对应着 没有声音 低杂音 中杂音 高杂音
    num_murmur3_classes = len(murmur3_classes)

    num_murmur_classes = len(murmur_classes)
    outcome_classes = ["Abnormal", "Normal"]
    # 这里定义了输出的结果的分类将声音分为了正常和不正常
    num_outcome_classes = len(outcome_classes)
    murmurmost_classes = ["AV", "MV", "PV", "TV"]
    num_murmurmost_classes = len(murmurmost_classes)
    murmurlocation_classes = ["AV", "MV", "PV", "TV"]
    num_murmurlocation_classes = len(murmurlocation_classes)
    features = list()
    murmurs = list()
    outcomes = list()
    murmurmosts = list()
    murmurlocations = list()
    # num_patient_files
    for i in tqdm(range(num_patient_files)):
        # 设置了进度条将range的进度可视化
        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        # 定义了一个临时的数组，patient数组中的第i个 ，然后使用load函数将文件内容读入临时数组中


        current_murmurlocation = np.zeros(num_murmurmost_classes, dtype=int)
        murmurlocation = get_murmurlocation(current_patient_data)
        murmurlocation = murmurlocation.split("+")
        # print(murmurlocation)
        murmurlocation=np.vstack(murmurlocation)
        for k in range(len(murmurlocation)):
            murmurlocation1 = murmurlocation[k]
            if murmurlocation1 in murmurlocation_classes:
                # 如果杂音的信息存在在现有的三个中
                j = murmurlocation_classes.index(murmurlocation1)
                # print(j) # 这里j存储的是murmur在数组中的序号 0 为"Present", 1 "Unknown", 2 "Absent"
                current_murmurlocation[j] = 1
        murmurlocations.append(current_murmurlocation)
        # print(murmurlocations)
        # 上述代码是自己添加的一个murmurlocation数组初始化，比如存在pv tv 那么 该序号上的数存为1 不存在则为0


        current_murmurmost = np.zeros(num_murmurmost_classes, dtype=int)
        murmurmost = get_murmurmost(current_patient_data)
        if murmurmost in murmurmost_classes:
            # 如果杂音的信息存在在现有的三个中
            j = murmurmost_classes.index(murmurmost)
            # print(j) # 这里j存储的是murmur在数组中的序号 0 为"Present", 1 "Unknown", 2 "Absent"
            current_murmurmost[j] = 1
        murmurmosts.append(current_murmurmost)
        # print(murmurmosts)
        # 在这里将杂音最多的位置存储在murmurmosts数组中，那个位置为1那个位置就是最多的

        # print(murmurs) 在这里将杂音的种类存储在murmurs数组中 看哪个位置有1即为这个种类
        # Outcome
        # 获取杂音最多的位置
        # Extract features.
        current_features = get_metadata(current_patient_data)
        # print(current_features)
        # print("++++++++++++")
        # 定义临时特征变量，获取文本中的信息，信息都有年龄 ，性别， 身高 ，体重 ，是否怀孕
        current_features = np.insert(
            current_features, 0, current_patient_data.split(" ")[0]
        )
        # print(current_features)
        # a=np.insert(arr, obj, values, axis)
        # arr原始数组，可一可多，obj插入元素位置，values是插入内容，axis是按行按列插入（0：行、1：列）。
        current_features = np.insert(
            current_features, 1, current_patient_data.split(" ")[2][:-3]
        )
        # print("------------------")
        # print(current_features)
        features.append(current_features)
        # 将刚刚提取到的个人数据添加到features列表中
        # Extract labels and use one-hot encoding.
        # Murmur
        current_murmur = np.zeros(num_murmur_classes, dtype=int)
        # 定义 存储三个数的数组
        murmur = get_murmur(current_patient_data)
        # 此时current_patient_data 中存储的是txt文件的全部信息
        if murmur in murmur_classes:
            # 如果杂音的信息存在在现有的三个中
            j = murmur_classes.index(murmur)
            # print(j) # 这里j存储的是murmur在数组中的序号 0 为"Present", 1 "Unknown", 2 "Absent"
            current_murmur[j] = 1
        murmurs.append(current_murmur)
        # print(murmurs) 在这里将杂音的种类存储在murmurs数组中 看哪个位置有1即为这个种类
        # Outcome
        current_outcome = np.zeros(num_outcome_classes, dtype=int)
        # 开了一个初值为0的二位的数组
        # 重点看
        outcome = get_outcome(current_patient_data)
        # 获取当前文件的outcome的值 abnormal 或者normal
        if outcome in outcome_classes:
            j = outcome_classes.index(outcome)
            current_outcome[j] = 1
        outcomes.append(current_outcome)
        # 此处跟之前murmur的处理基本一样
    features = np.vstack(features)
    # 将原先的元组 由原先的水平方向上堆叠 ，变为竖直方向上堆叠
    murmurs = np.vstack(murmurs)
    outcomes = np.vstack(outcomes)
    murmurmosts = np.vstack(murmurmosts)
    murmurlocations = np.vstack(murmurlocations)
    # print(murmurmosts)
    # 将原先的元组 由原先的水平方向上堆叠 ，变为竖直方向上堆叠
    # Combine dataframes
    # print(features)
    features_pd = pd.DataFrame(
        features,
        columns=[
            "id",
            "hz",
            "age",
            "female",
            "male",
            "height",
            "weight",
            "is_pregnant",
        ],
    )
    # 以上函数使用pd.dataframe 将原本竖直的数组 放入一个表格里面 按照column 的序列排序 分别是ID hz 年龄等等
    # print(features_pd)
    murmurs_pd = pd.DataFrame(murmurs, columns=murmur_classes)
    outcomes_pd = pd.DataFrame(outcomes, columns=outcome_classes)
    # 此处也是跟以上同理，将数据放入表格中
    murmurmosts_pd = pd.DataFrame(murmurmosts, columns=murmurmost_classes)
    murmurlocations_pd = pd.DataFrame(murmurlocations,columns=murmurlocation_classes)
    complete_pd = pd.concat([features_pd, murmurs_pd, outcomes_pd, murmurmosts_pd,murmurlocations_pd], axis=1)
    # print(complete_pd) 此处将三个表格联合在一起，组成一个混合的表格
    complete_pd["id"] = complete_pd["id"].astype(int).astype(str)
    #  上述代码使用强制类型转换，将id的值变为int   print(complete_pd["id"][1])
    # Split data
    complete_pd["stratify_column"] = (
        complete_pd[stratified_features].astype(str).agg("-".join, axis=1)
    )
    # print(complete_pd["stratify_column"][i])
    # print("aaaaa")
    complete_pd_train, complete_pd_test = train_test_split(
        complete_pd,
        test_size=test_size,
        random_state=random_state,
        stratify=complete_pd["stratify_column"],
    )
    # train_test_split函数功能为根据比例随机地从数据集中挑选test 和train
    # 根据输入的test_size 定义test数据集的大小 以及训练数据集的大小 vali_size 失败值为0.16 test_size为0.2
    vali_split = vali_size / (1 - test_size)
    complete_pd_train, complete_pd_val = train_test_split(
        complete_pd_train,
        test_size=vali_split,
        random_state=random_state + 1,
        stratify=complete_pd_train["stratify_column"],
    )
    # cross_validation.train_test_split(train_data,train_target,test_size=0.4, random_state=0)
    #
    # 参数解释：
    # train_data：所要划分的样本特征集
    #
    # train_target：所要划分的样本结果
    #
    # test_size：样本占比，如果是整数的话就是样本的数量
    #
    # random_state：是随机数的种子。
    with open(os.path.join(out_directory, "split_details.txt"), "w") as text_file:
        text_file.write("This data split is stratified over the following features: \n")
        for feature in stratified_features:
            text_file.write(feature + ", ")
    # Save the files.
    os.makedirs(os.path.join(out_directory, "train_data"))
    os.makedirs(os.path.join(out_directory, "vali_data"))
    os.makedirs(os.path.join(out_directory, "test_data"))
    # 分别创建三个文件夹 来存储三个分开的数据集
    for f in complete_pd_train["id"]:
        copy_files(
            data_directory,
            f,
            os.path.join(out_directory, "train_data/"),
        )
        # 将文件从源文件夹读入到新文件夹
    for f in complete_pd_val["id"]:
        copy_files(
            data_directory,
            f,
            os.path.join(out_directory, "vali_data/"),
        )
    for f in complete_pd_test["id"]:
        copy_files(
            data_directory,
            f,
            os.path.join(out_directory, "test_data/"),
        )


def copy_files(data_directory: str, ident: str, out_directory: str) -> None:
    # Get the list of files in the data folder.
    files = os.listdir(data_directory)
    # Copy all files in data_directory that start with f to out_directory
    for f in files:
        if f.startswith(ident):
            _ = shutil.copy(os.path.join(data_directory, f), out_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="StratifiedDataSplit")
    parser.add_argument(
        "--data_directory",
        type=str,
        help="The directory containing the data you wish to split.",
        default="training_data",
    )
    parser.add_argument(
        "--out_directory",
        type=str,
        help="The directory to store the split data.",
        default="data/stratified_data",
    )
    parser.add_argument(
        "--vali_size", type=float, default=0.16, help="The size of the test split."
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="The size of the test split."
    )
    parser.add_argument(
        "--random_state", type=int, default=5678, help="The random state for the split."
    )
    args = parser.parse_args()

    stratified_features = ["Normal", "Abnormal", "Absent", "Present", "Unknown"]

    # Create the test split.
    stratified_test_vali_split(stratified_features, **vars(args))
