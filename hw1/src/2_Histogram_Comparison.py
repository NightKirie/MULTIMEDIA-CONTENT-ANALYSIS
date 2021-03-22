import os
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import math
import collections
from Ground_Truth import *

def Histogram_Comparison_Gray_1D(file_list, t):
    img_1 = cv2.cvtColor(cv2.imread(file_list[0]), cv2.COLOR_BGR2GRAY).astype("int16")
    img_1_hist = np.bincount(img_1.reshape(img_1.shape[0] * img_1.shape[1]), minlength=256)
    shot_change_index = []
    sd_list = []
    for i in range(1, len(file_list)-1):
        img_2 = cv2.cvtColor(cv2.imread(file_list[i]), cv2.COLOR_BGR2GRAY).astype("int16")
        img_2_hist = np.bincount(img_2.reshape(img_2.shape[0] * img_2.shape[1]), minlength=256)
        sd = np.sum(abs(img_1_hist - img_2_hist))
        img_1_hist = img_2_hist
        if sd > t:
            shot_change_index.append(i)
        sd_list.append(sd)
    return (shot_change_index, sd_list)

def Histogram_Comparison_Gray_2D(file_list, t):
    img_1 = cv2.cvtColor(cv2.imread(file_list[0]), cv2.COLOR_BGR2GRAY).astype("int16")
    img_1_hist = np.bincount(img_1.reshape(img_1.shape[0] * img_1.shape[1]), minlength=256)
    shot_change_index = []
    sd_list = []
    for i in range(1, len(file_list)-1):
        img_2 = cv2.cvtColor(cv2.imread(file_list[i]), cv2.COLOR_BGR2GRAY).astype("int16")
        img_2_hist = np.bincount(img_2.reshape(img_2.shape[0] * img_2.shape[1]), minlength=256)
        sd = 0
        for j in range(0, 256):
            sd += ((img_1_hist[j] - img_2_hist[j]) ** 2) / img_2_hist[j] if img_2_hist[j] != 0 else 0
        img_1_hist = img_2_hist
        if sd > t:
            shot_change_index.append(i)
        sd_list.append(sd)
    return (shot_change_index, sd_list)

def count_subarray(arr):
    arr = arr.reshape(-1, arr.shape[-1])
    tuple_arr = [tuple(x) for x in arr]
    return collections.Counter(tuple_arr)


def Histogram_Comparison_RGB(file_list, t):
    shot_change_index = []
    chd_list = []
    img_1 = cv2.imread(file_list[0]).astype("int16")
    img_1 = img_1 // 64
    n = img_1.shape[0] * img_1.shape[1]
    img_1_p = count_subarray(img_1)
    for i in range(1, len(file_list)-1):
        img_2 = cv2.imread(file_list[i]).astype("int16")
        img_2 = img_2 // 64
        img_2_p = count_subarray(img_2)
        chd = (sum((img_1_p - img_2_p).values()) + sum((img_2_p - img_1_p).values())) / n
        if chd > t:
            shot_change_index.append(i)
        chd_list.append(chd)
        img_1_p = img_2_p
    return (shot_change_index, chd_list)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # news
    news_file_list = []
    for dir_path, _, file_list in os.walk("../news_out"):
        news_file_list = [os.path.join(dir_path, file) for file in file_list]
    with open("../result/2/news.txt", "w+") as f:
        print("Working on news 1")
        (shot_change_index, sd_list) = Histogram_Comparison_Gray_1D(news_file_list, math.sqrt(352*240)*75)
        f.write("Histogram comparison with gray color & default equation for shot-change detection of news.mpg:\n")
        f.write(f"{shot_change_index}\n\n")
        plt.plot(sd_list)
        plt.plot(NEWS_ANS, [0]*len(NEWS_ANS), "ro")
        plt.savefig("../result/2/news_1.jpg")
        plt.clf()
        
        print("Working on news 2")
        (shot_change_index, sd_list) = Histogram_Comparison_Gray_2D(news_file_list, math.sqrt(352*240)*50)
        f.write("Histogram comparison with gray color & X2-test for shot-change detection of news.mpg:\n")
        f.write(f"{shot_change_index}\n\n")
        plt.plot(sd_list)
        plt.plot(NEWS_ANS, [0]*len(NEWS_ANS), "ro")
        plt.savefig("../result/2/news_2.jpg")
        plt.clf()
        
        print("Working on news 3")
        (shot_change_index, chd_list) = Histogram_Comparison_RGB(news_file_list, 0.25)
        f.write("Histogram comparison with rgb color for shot-change detection of news.mpg:\n")
        f.write(f"{shot_change_index}\n\n")
        plt.plot(chd_list)
        plt.plot(NEWS_ANS, [0]*len(NEWS_ANS), "ro")
        plt.savefig("../result/2/news_3.jpg")
        plt.clf()

    # soccer
    soccer_file_list = []
    for dir_path, _, file_list in os.walk("../soccer_out"):
        soccer_file_list = [os.path.join(dir_path, file) for file in file_list]
    with open("../result/2/soccer.txt", "w+") as f:
        print("Working on soccer 1")
        (shot_change_index, sd_list) = Histogram_Comparison_Gray_1D(soccer_file_list, math.sqrt(360*240)*75)
        f.write("Histogram comparison with gray color & default equation for shot-change detection of soccer.mpg:\n")
        f.write(f"{shot_change_index}\n\n")
        plt.plot(sd_list)
        plt.plot(SOCCER_ANS, [0]*len(SOCCER_ANS), "ro")
        plt.savefig("../result/2/soccer_1.jpg")
        plt.clf()
        
        print("Working on soccer 2")
        (shot_change_index, sd_list) = Histogram_Comparison_Gray_2D(soccer_file_list, math.sqrt(360*240)*50)
        f.write("Histogram comparison with gray color & X2-test for shot-change detection of soccer.mpg:\n")
        f.write(f"{shot_change_index}\n\n")
        plt.plot(sd_list)
        plt.plot(SOCCER_ANS, [0]*len(SOCCER_ANS), "ro")
        plt.savefig("../result/2/soccer_2.jpg")
        plt.clf()
        
        print("Working on soccer 3")
        (shot_change_index, chd_list) = Histogram_Comparison_RGB(soccer_file_list, 0.25)
        f.write("Histogram comparison with rgb color for shot-change detection of soccer.mpg:\n")
        f.write(f"{shot_change_index}\n\n")
        plt.plot(chd_list)
        plt.plot(SOCCER_ANS, [0]*len(SOCCER_ANS), "ro")
        plt.savefig("../result/2/soccer_3.jpg")
        plt.clf()
    
    # ngc
    ngc_file_list = []
    for dir_path, _, file_list in os.walk("../ngc_out"):
        ngc_file_list = [os.path.join(dir_path, file) for file in file_list]
    with open("../result/2/ngc.txt", "w+") as f:
        print("Working on ngc 1")
        (shot_change_index, sd_list) = Histogram_Comparison_Gray_1D(ngc_file_list, math.sqrt(864*480)*75)
        f.write("Histogram comparison with gray color & default equation for shot-change detection of ngc.mpg:\n")
        f.write(f"{shot_change_index}\n\n")
        plt.plot(sd_list)
        plt.plot(NGC_ANS, [0]*len(NGC_ANS), "ro")
        plt.savefig("../result/2/ngc_1.jpg")
        plt.clf()
        
        print("Working on ngc 2")
        (shot_change_index, sd_list) = Histogram_Comparison_Gray_2D(ngc_file_list, math.sqrt(864*480)*50)
        f.write("Histogram comparison with gray color & X2-test for shot-change detection of ngc.mpg:\n")
        f.write(f"{shot_change_index}\n\n")
        plt.plot(sd_list)
        plt.plot(NGC_ANS, [0]*len(NGC_ANS), "ro")
        plt.savefig("../result/2/ngc_2.jpg")
        plt.clf()
        
        print("Working on ngc 3")
        (shot_change_index, chd_list) = Histogram_Comparison_RGB(ngc_file_list, 0.1)
        f.write("Histogram comparison with rgb color for shot-change detection of ngc.mpg:\n")
        f.write(f"{shot_change_index}\n\n")
        plt.plot(chd_list)
        plt.plot(NGC_ANS, [0]*len(NGC_ANS), "ro")
        plt.savefig("../result/2/ngc_3.jpg")
        plt.clf()
