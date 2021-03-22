import os
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import math
import collections
from Ground_Truth import *

def Likelihood_Ratio(file_list, large_num, n, t=2.0):
    img_1 = cv2.cvtColor(cv2.imread(file_list[0]), cv2.COLOR_BGR2GRAY)
    large_num_list = []
    large_block_ratio_list = []
    for i in range(1, len(file_list)-1):
        img_2 = cv2.cvtColor(cv2.imread(file_list[i]), cv2.COLOR_BGR2GRAY)
        block_ratio_list = np.empty(0)
        for x in range(0, img_1.shape[0], n):
            for y in range(0, img_1.shape[1], n):
                m1 = np.mean(img_1[x:x+n, y:y+n])
                m2 = np.mean(img_2[x:x+n, y:y+n])
                s1 = np.var(img_1[x:x+n, y:y+n])
                s2 = np.var(img_2[x:x+n, y:y+n])
                block_ratio_list = np.append(block_ratio_list, (((s1 + s2)/2 + ((m1 - m2)/2)**2)**2)/(s1*s2))
        # print(block_ratio_list)
        large_block_ratio = np.count_nonzero(block_ratio_list > 2.0)
        if large_block_ratio > large_num:
            large_num_list.append(i)
        large_block_ratio_list.append(large_block_ratio)
        img_1 = img_2
    return (large_num_list, large_block_ratio_list)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # news
    # news_file_list = []
    # for dir_path, _, file_list in os.walk("../news_out"):
    #     news_file_list = [os.path.join(dir_path, file) for file in file_list]
    # with open("../result/3/news.txt", "w+") as f:
    #     print("Working on news")
    #     (large_num_list, large_block_ratio_list) = Likelihood_Ratio(news_file_list, (352/8)*(240/8)*0.5, 8)
    #     f.write("likelihood ratio with gray color for shot-change detection of news.mpg:\n")
    #     f.write(f"{large_num_list}\n\n")
    #     plt.plot(large_block_ratio_list)
    #     plt.plot(NEWS_ANS, [0]*len(NEWS_ANS), "ro")
    #     plt.savefig("../result/3/news.jpg")
    #     plt.clf()
    
    # soccer
    soccer_file_list = []
    for dir_path, _, file_list in os.walk("../soccer_out"):
        soccer_file_list = [os.path.join(dir_path, file) for file in file_list]
    with open("../result/3/soccer.txt", "w+") as f:
        print("Working on soccer")
        (large_num_list, large_block_ratio_list) = Likelihood_Ratio(soccer_file_list, (360/8)*(240/8)*0.45, 8)
        f.write("likelihood ratio with gray color for shot-change detection of soccer.mpg:\n")
        f.write(f"{large_num_list}\n\n")
        plt.plot(large_block_ratio_list)
        plt.plot(SOCCER_ANS, [0]*len(SOCCER_ANS), "ro")
        plt.savefig("../result/3/soccer.jpg")
        plt.clf()
    

    # ngc
    # ngc_file_list = []
    # for dir_path, _, file_list in os.walk("../ngc_out"):
    #     ngc_file_list = [os.path.join(dir_path, file) for file in file_list]
    # with open("../result/3/ngc.txt", "w+") as f:
    #     print("Working on ngc")
    #     (large_num_list, large_block_ratio_list) = Likelihood_Ratio(ngc_file_list, (864/8)*(480/8)*0.3, 8)
    #     f.write("likelihood ratio with gray color for shot-change detection of ngc.mpg:\n")
    #     f.write(f"{large_num_list}\n\n")
    #     plt.plot(large_block_ratio_list)
    #     plt.plot(NGC_ANS, [0]*len(NGC_ANS), "ro")
    #     plt.savefig("../result/3/ngc.jpg")
    #     plt.clf()