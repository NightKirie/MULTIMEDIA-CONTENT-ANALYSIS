import os
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import math
import collections
from Ground_Truth import *

def Edge_Detection(file_list, t):
    entering_ratio = []
    exiting_ratio = []
    shot_change_list = []
    img_1 = cv2.GaussianBlur(cv2.cvtColor(cv2.imread(file_list[0]), cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edge_1 = cv2.Canny(img_1, 10, 100)
    for i in range(1, len(file_list)-1):
        img_2 = cv2.GaussianBlur(cv2.cvtColor(cv2.imread(file_list[i]), cv2.COLOR_BGR2GRAY), (5, 5), 0)        
        edge_2 = cv2.Canny(img_2, 10, 100)
        entering_edge = edge_2 - edge_1 
        exiting_edge = edge_1 - edge_2
        x_in = collections.Counter(entering_edge.reshape(-1))[255]
        x_out = collections.Counter(exiting_edge.reshape(-1))[255]
        z_1 = collections.Counter(edge_1.reshape(-1))[255]
        z_2 = collections.Counter(edge_2.reshape(-1))[255]
        if z_1 != 0 and z_2 != 0:
            entering_ratio.append(x_in/z_2)
            exiting_ratio.append(x_out/z_1)
            if max(x_in/z_2, x_out/z_1) > t:
                shot_change_list.append(i)
        else:
            entering_ratio.append(0)
            exiting_ratio.append(0)
        edge_1 = edge_2
    return (entering_ratio, exiting_ratio, shot_change_list)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # news
    news_file_list = []
    for dir_path, _, file_list in os.walk("../news_out"):
        news_file_list = [os.path.join(dir_path, file) for file in file_list]
    print("Working on news")
    with open("../result/4/news.txt", "w+") as f:
        (entering_ratio, exiting_ratio, shot_change_list) = Edge_Detection(news_file_list, 0.55)
        f.write("Edge change ratio for shot-change detection of news.mpg:\n")
        f.write(f"{shot_change_list}\n\n")
        plt.plot(entering_ratio, "bx")
        plt.plot(exiting_ratio, "g.")
        plt.plot(NEWS_ANS, [0]*len(NEWS_ANS), "ro")
        plt.savefig("../result/4/news_1.jpg")
        plt.clf()
        plt.plot(np.maximum(entering_ratio, exiting_ratio))
        plt.plot(NEWS_ANS, [0]*len(NEWS_ANS), "ro")
        plt.savefig("../result/4/news_2.jpg")
        plt.clf()
    
    # soccer
    soccer_file_list = []
    for dir_path, _, file_list in os.walk("../soccer_out"):
        soccer_file_list = [os.path.join(dir_path, file) for file in file_list]
    print("Working on soccer")
    with open("../result/4/soccer.txt", "w+") as f:
        (entering_ratio, exiting_ratio, shot_change_list) = Edge_Detection(soccer_file_list, 0.68)
        f.write("Edge change ratio for shot-change detection of soccer.mpg:\n")
        f.write(f"{shot_change_list}\n\n")
        plt.plot(entering_ratio, "bx")
        plt.plot(exiting_ratio, "g.")
        plt.plot(SOCCER_ANS, [0]*len(SOCCER_ANS), "ro")
        plt.savefig("../result/4/soccer_1.jpg")
        plt.clf()
        plt.plot(np.maximum(entering_ratio, exiting_ratio))
        plt.plot(SOCCER_ANS, [0]*len(SOCCER_ANS), "ro")
        plt.savefig("../result/4/soccer_2.jpg")
        plt.clf()
    

    # ngc
    ngc_file_list = []
    for dir_path, _, file_list in os.walk("../ngc_out"):
        ngc_file_list = [os.path.join(dir_path, file) for file in file_list]
    print("Working on ngc")
    with open("../result/4/ngc.txt", "w+") as f:
        (entering_ratio, exiting_ratio, shot_change_list) = Edge_Detection(ngc_file_list, 0.8)
        f.write("Edge change ratio for shot-change detection of ngc.mpg:\n")
        f.write(f"{shot_change_list}\n\n")
        plt.plot(entering_ratio, "bx")
        plt.plot(exiting_ratio, "g.")
        plt.plot(NGC_ANS, [0]*len(NGC_ANS), "ro")
        plt.savefig("../result/4/ngc_1.jpg")
        plt.clf()
        plt.plot(np.maximum(entering_ratio, exiting_ratio))
        plt.plot(NGC_ANS, [0]*len(NGC_ANS), "ro")
        plt.savefig("../result/4/ngc_2.jpg")
        plt.clf()