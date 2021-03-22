import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from Ground_Truth import *


"""
    t: threshold for gray value difference
    cp: pixel change percentage for gray value difference over threshold
"""
def Pair_Wise_Comparison(file_list, t=15, cp=0.4):
    img_1 = cv2.cvtColor(cv2.imread(file_list[0]), cv2.COLOR_BGR2GRAY).astype("int16")
    img_pixel = img_1.shape[0] * img_1.shape[1]
    shot_change_index = []
    dp_list = []
    for i in range(1, len(file_list)-1):
        img_2 = cv2.cvtColor(cv2.imread(file_list[i]), cv2.COLOR_BGR2GRAY).astype("int16")
        dp = (abs(img_1 - img_2) > t).sum()
        if dp > (img_pixel * cp):
            shot_change_index.append(i)
        dp_list.append(dp)
        img_1 = img_2   

    return (shot_change_index, dp_list)

"""
    t: threshold for gray value difference
    cp: pixel change percentage for gray value difference over threshold
    w: window size for smoothing comparison
"""
def Pair_Wise_Comparison_Window(file_list, t=15, cp=0.4, w=3):
    img_1 = cv2.blur(cv2.cvtColor(cv2.imread(file_list[0]), cv2.COLOR_BGR2GRAY), (w, w)).astype("int16")
    img_pixel = img_1.shape[0] * img_1.shape[1]
    shot_change_index = []
    dp_list = []
    for i in range(1, len(file_list)-1):
        img_2 = cv2.blur(cv2.cvtColor(cv2.imread(file_list[i]), cv2.COLOR_BGR2GRAY), (w, w)).astype("int16")
        dp = (abs(img_1 - img_2) > t).sum()
        if dp > (img_pixel * cp):
            shot_change_index.append(i)
        dp_list.append(dp)
        img_1 = img_2   

    return (shot_change_index, dp_list)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # news
    news_file_list = []
    for dir_path, _, file_list in os.walk("../news_out"):
        news_file_list = [os.path.join(dir_path, file) for file in file_list]
    with open("../result/1/news.txt", "w+") as f:
        print("Working on news 1")
        (shot_change_index, dp_list) = Pair_Wise_Comparison(news_file_list)
        f.write("Default pair-wise comparison for shot-change detection of news.mpg:\n")
        f.write(f"{shot_change_index}\n\n")
        plt.plot(dp_list)
        plt.plot(NEWS_ANS, [0]*len(NEWS_ANS), "ro")
        plt.savefig("../result/1/news_1.jpg")
        plt.clf()

        print("Working on news 2")
        (shot_change_index, dp_list) = Pair_Wise_Comparison_Window(news_file_list)
        f.write("Windowed pair-wise comparison for shot-change detection of news.mpg:\n")
        f.write(f"{shot_change_index}\n\n")
        plt.plot(dp_list)
        plt.plot(NEWS_ANS, [0]*len(NEWS_ANS), "ro")
        plt.savefig("../result/1/news_2.jpg")
        plt.clf()

    # soccer
    soccer_file_list = []
    for dir_path, _, file_list in os.walk("../soccer_out"):
        soccer_file_list = [os.path.join(dir_path, file) for file in file_list]
    with open("../result/1/soccer.txt", "w+") as f:
        print("Working on soccer 1")
        (shot_change_index, dp_list) = Pair_Wise_Comparison(soccer_file_list)
        f.write("Default pair-wise comparison for shot-change detection of soccer.mpg:\n")
        f.write(f"{shot_change_index}\n\n")
        plt.plot(dp_list)
        plt.plot(SOCCER_ANS, [0]*len(SOCCER_ANS), "ro")
        plt.savefig("../result/1/soccer_1.jpg")
        plt.clf()

        print("Working on soccer 2")
        (shot_change_index, dp_list) = Pair_Wise_Comparison_Window(soccer_file_list)
        f.write("Windowed pair-wise comparison for shot-change detection of soccer.mpg:\n")
        f.write(f"{shot_change_index}\n\n")
        plt.plot(dp_list)
        plt.plot(SOCCER_ANS, [0]*len(SOCCER_ANS), "ro")
        plt.savefig("../result/1/soccer_2.jpg")
        plt.clf()

    # ngc
    ngc_file_list = []
    for dir_path, _, file_list in os.walk("../ngc_out"):
        ngc_file_list = [os.path.join(dir_path, file) for file in file_list]
    with open("../result/1/ngc.txt", "w+") as f:
        print("Working on ngc 1")
        (shot_change_index, dp_list) = Pair_Wise_Comparison(ngc_file_list)
        f.write("Default pair-wise comparison for shot-change detection of ngc.mpg:\n")
        f.write(f"{shot_change_index}\n\n")
        plt.plot(dp_list)
        plt.plot(NGC_ANS, [0]*len(NGC_ANS), "ro")
        plt.savefig("../result/1/ngc_1.jpg")
        plt.clf()

        print("Working on ngc 2")
        (shot_change_index, dp_list) = Pair_Wise_Comparison_Window(ngc_file_list)
        f.write("Windowed pair-wise comparison for shot-change detection of ngc.mpg:\n")
        f.write(f"{shot_change_index}\n\n")
        plt.plot(dp_list)
        plt.plot(NGC_ANS, [0]*len(NGC_ANS), "ro")
        plt.savefig("../result/1/ngc_2.jpg")
        plt.clf()
