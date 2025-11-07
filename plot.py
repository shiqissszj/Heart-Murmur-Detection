import numpy as np
import librosa
from sklearn.ensemble import RandomForestRegressor
import wave
import numpy as np
import os
import shutil
import librosa.display
import librosa
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.io import wavfile
from tqdm import tqdm
import random
from find_and_load_patient_files import (
    find_wav_files,
    load_patient_data,
)
from tqdm import tqdm

def cut(input_path, out_put, cut_length):
    # 打开wav文件 ，open返回一个的是一个Wave_read类的实例，通过调用它的方法读取WAV文件的格式和数据。
    f = wave.open(input_path, "rb")

    # 读取格式信息
    # 一次性返回所有的WAV文件的格式信息，它返回的是一个组元(tuple)：声道数, 量化位数（byte单位）, 采样频率, 采样点数, 压缩类型, 压缩类型的描述。wave模块只支持非压缩的数据，因此可以忽略最后两个信息
    params = f.getparams()

    nchannels, sampwidth, framerate, nframes = params[:4]

    print("你要分割的音频所有参数参数为:")
    print("通道数：%d,量化位数：%d,采样频率：%d,帧数(采样点数)：%d" % (nchannels, sampwidth * 8, framerate, nframes))

    # 读取波形数据
    # 读取声音数据，传递一个参数指定需要读取的长度（以取样点为单位）
    str_data = f.readframes(nframes)
    f.close()

    # 将波形数据转换成数组
    # 需要根据声道数和量化单位，将读取的二进制数据转换为一个可以计算的数组
    wave_data = np.frombuffer(str_data, dtype=np.int16)
    # print(wave_data.shape)

    # 将wave_data数组改为2列，行数自动匹配。在修改shape的属性时，需使得数组的总长度不变。
    wave_data.shape = -1, nchannels
    # wave_data.shape = 2, -1 # 这种情况要先转置
    # print(wave_data)
    # 转置数据
    wave_data = wave_data.T
    # print(wave_data)

    # 切割帧数
    # cut_length = 10*framerate
    print("切割的固定长度为:%d" % cut_length)

    i = 1
    for num in range(0, nframes // cut_length * cut_length, cut_length):  # 保证每一段够长，不够长舍掉
        print(num)
        # 切片
        now_wave_data = wave_data[:, num:num + cut_length]  # 切片不包括后端点
        print(now_wave_data.shape)
        # 打开WAV文档
        f = wave.open(out_put + str(i) + ".wav", "wb")
        # 配置声道数、量化位数和取样频率
        f.setnchannels(nchannels)
        f.setsampwidth(sampwidth)
        f.setframerate(framerate)
        # 将wav_data转换为二进制数据写入文件
        f.writeframes(now_wave_data.tobytes())
        f.close()
        # f = wave.open(out_put, "rb")
        f = wave.open(out_put + str(i) + ".wav", "rb")
        params = f.getparams()
        print(params[:4])
        f.close()
        i += 1


def displayWaveform(data_directory, out_directory):  # 显示语音时域波形
    """
    display waveform of a given speech sample
    :param sample_name: speech sample name
    :param fs: sample frequency
    :return:
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Show Chinese label
    plt.rcParams['axes.unicode_minus'] = False  # These two lines need to be set manually
    cnt = 1
    patient_files = find_wav_files(data_directory)
    for filename in patient_files:
        y, sr = librosa.load(filename, sr=16000)

        # 定义窗口长度和重叠比例
        win_length = 1024
        hop_length = int(win_length / 2)

        # 对音频数据进行滑动窗口切割
        n_frames = 1 + int((len(y) - win_length) / hop_length)
        y_frames = np.zeros((n_frames, win_length))
        for i in range(n_frames):
            y_frames[i, :] = y[i * hop_length: i * hop_length + win_length]

        # 提取每个窗口的特征
        features = []
        for frame in y_frames:
            # 这里以频域特征为例，计算每个窗口的频谱图，并提取频域特征
            spec = np.abs(librosa.stft(frame))
            feature = np.mean(spec, axis=1)
            features.append(feature)
        features = np.array(features)

        # 定义标签，这里以能量大小作为标签
        labels = np.sum(np.square(y_frames), axis=1)

        # 使用随机森林模型进行特征重要性预测
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(features, labels)
        importances = rf.feature_importances_

        # 输出特征重要性
        print('Feature Importances:', importances)
        # 计算每个窗口的影响力
        imp_frames = rf.predict(features)

        # 将特征值乘以影响力
        feat_frames = features * imp_frames.reshape(-1, 1)

        # 计算每个采样点的平均影响力
        imp_samples = np.zeros(len(y))
        count_samples = np.zeros(len(y))
        for i in tqdm(range(n_frames - 1)):
            imp_samples[i * hop_length: i * hop_length + win_length + 1] += feat_frames[i, :]
            count_samples[i * hop_length: i * hop_length + win_length] += 1
        imp_samples /= count_samples

        # 输出每个采样点的平均影响力
        print('Sample Impacts:', imp_samples)
        # print(len(imp_samples))
        # print(max(imp_samples))

        plt.rcParams['font.sans-serif'] = ['SimHei']  # Show Chinese label
        plt.rcParams['axes.unicode_minus'] = False  # These two lines need to be set manually

        samples, sr = librosa.load(filename, sr=16000)
        # samples = samples[6000:16000]
        # print(len(samples), sr)
        time = np.arange(0, len(samples)) * (1.0 / sr)
        # print(samples)

        plt.figure(figsize=(20, 5))
        weights = [random.randint(0, 9) for _ in range(len(time))]
        weights = imp_samples
        # 定义颜色映射表
        cmap = plt.cm.get_cmap('coolwarm')
        # print(len(time) - 1)
        # print(weights)
        # 绘制折线图和数据点
        for i in tqdm(range(len(imp_samples) - 1)):
            # 确定当前区域的权值和颜色
            w = imp_samples[i]
            color = cmap(w)
            # print(color)

            # 绘制折线
            plt.plot([time[i], time[i + 1]], [samples[i], samples[i + 1]], color = 'black', linewidth=2)
            # color=color
            # 绘制数据点 语音信号时域波形
            # plt.scatter(time[i], samples[i], color=color, s=50)
        plt.title("Time domain waveform of speech signal")
        plt.xlabel("Duration (seconds)")
        plt.ylabel("amplitude")
        print(out_directory + cnt.__str__() + ".png")
        plt.savefig(out_directory + cnt.__str__() + ".png", dpi=1600)
        cnt = cnt + 1
        plt.show()


if __name__ == '__main__':
    # displayWaveform("1/", "1/")
    cut_length = 1024
    data_directory = "../data_training"
    patient_files1 = find_wav_files(data_directory)
    for filename in patient_files1:
        # print(filename)
        new_string = filename[17:]
        # print(new_string)
        if os.path.exists(new_string):
            shutil.rmtree(new_string)
        else:
            os.makedirs(new_string)

        cut(filename, new_string+"/", cut_length)
        displayWaveform(new_string+"/", new_string+"/")



