import os
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import math
import collections

def Histogram_Comparison_Gray(file_list, t):
    img_1 = cv2.cvtColor(cv2.imread(file_list[0]), cv2.COLOR_BGR2GRAY).astype("int16")
    img_1_hist = np.bincount(img_1.reshape(img_1.shape[0] * img_1.shape[1]), minlength=256)
    shot_change_index = []
    sd_list = []
    for i in range(1, len(file_list)-1):
        img_2 = cv2.cvtColor(cv2.imread(file_list[i]), cv2.COLOR_BGR2GRAY).astype("int16")
        img_2_hist = np.bincount(img_2.reshape(img_2.shape[0] * img_2.shape[1]), minlength=256)
        # sd = np.sum(((img_1_hist - img_2_hist) ** 2) / img_2_hist) # bad result
        sd = np.sum(abs(img_1_hist - img_2_hist))
        img_1_hist = img_2_hist
        if sd > t:
            shot_change_index.append(i)
        sd_list.append(sd)
    plt.plot(sd_list)
    plt.show()
    return shot_change_index

def count_subarray(arr):
    arr = arr.reshape(-1, arr.shape[-1])
    tuple_arr = [tuple(x) for x in arr]
    return collections.Counter(tuple_arr)


def Histogram_Comparison_RGB(file_list):
    shot_change_index = []
    chd_list = []
    img_1 = cv2.imread(file_list[0]).astype("int16")
    n = img_1.shape[0] * img_1.shape[1]
    img_1_p = count_subarray(img_1)
    # img_1_hist_b = np.bincount(img_1[0].reshape(img_1.shape[0] * img_1.shape[1]), minlength=256)
    for i in range(1, len(file_list)-1):
        img_2 = cv2.imread(file_list[i]).astype("int16")
        img_2_p = count_subarray(img_2)
        chd = (sum((img_1_p - img_2_p).values()) + sum((img_2_p - img_1_p).values())) / n
        chd_list.append(chd)
        img_1_p = img_2_p
        
        print(i)
    # plt.plot(chd_list)
    # plt.show()
    
    return shot_change_index

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # news
    # news_file_list = []
    # for dir_path, _, file_list in os.walk("../news_out"):
        # news_file_list = [os.path.join(dir_path, file) for file in file_list]
    # print("Histogram comparison with gray color for shot-change detection of news.mpg: ", Histogram_Comparison_Gray(news_file_list, math.sqrt(352*240)*75))
    # print("Histogram comparison with rgb color for shot-change detection of news.mpg: ", Histogram_Comparison_RGB(news_file_list))

    # # soccer
    # soccer_file_list = []
    # list = []
    # for dir_path, _, file_list in os.walk("../soccer_out"):
    #     soccer_file_list = [os.path.join(dir_path, file) for file in file_list]
    # print("Histogram comparison with gray color for shot-change detection of soccer.mpg: ", Histogram_Comparison_Gray(soccer_file_list, math.sqrt(360*240)*75))
    # print("Histogram comparison with rgb color for shot-change detection of soccer.mpg: ", Histogram_Comparison_RGB(soccer_file_list))

    # print()

    # # ngc
    ngc_file_list = []
    for dir_path, _, file_list in os.walk("../ngc_out"):
        ngc_file_list = [os.path.join(dir_path, file) for file in file_list]
    # print("Histogram comparison with gray color for shot-change detection of ngc.mpg: ", Histogram_Comparison_Gray(ngc_file_list, math.sqrt(864*480)*75))
    print("Histogram comparison with rgb color for shot-change detection of ngc.mpg: ", Histogram_Comparison_RGB(ngc_file_list))
    # print()