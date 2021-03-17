import os
import numpy as np
import cv2 
import matplotlib.pyplot as plt

def Histogram_Comparison_Gray(file_list, t=10):
    img_1 = cv2.cvtColor(cv2.imread(file_list[0]), cv2.COLOR_BGR2GRAY).astype("int16")
    img_1_hist = np.bincount(img_1.reshape(img_1.shape[0] * img_1.shape[1]), minlength=256)
    shot_change_index = []
    for i in range(1, len(file_list)-1):
        img_2 = cv2.cvtColor(cv2.imread(file_list[i]), cv2.COLOR_BGR2GRAY).astype("int16")
        img_2_hist = np.bincount(img_2.reshape(img_2.shape[0] * img_2.shape[1]), minlength=256)
        # sd = np.sum(((img_1_hist - img_2_hist) ** 2) / img_2_hist)
        sd = np.sum(abs(img_1_hist - img_2_hist))
        img_1_hist = img_2_hist
        shot_change_index.append(sd)
    plt.plot(shot_change_index)
    plt.show()
    return shot_change_index

def Histogram_Comparison_Color(file_list):
    shot_change_index = []
    return shot_change_index

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # news
    news_file_list = []
    for dir_path, _, file_list in os.walk("../news_out"):
        news_file_list = [os.path.join(dir_path, file) for file in file_list]
    print("Histogram comparison for shot-change detection of news.mpg: ", Histogram_Comparison_Gray(news_file_list))
    print()

    # # soccer
    # soccer_file_list = []
    # for dir_path, _, file_list in os.walk("../soccer_out"):
    #     soccer_file_list = [os.path.join(dir_path, file) for file in file_list]
    # print("Histogram comparison for shot-change detection of news.mpg: ", Histogram_Comparison_Gray(soccer_file_list))
    # print()

    # # ngc
    # ngc_file_list = []
    # for dir_path, _, file_list in os.walk("../ngc_out"):
    #     ngc_file_list = [os.path.join(dir_path, file) for file in file_list]
    # print("Histogram comparison for shot-change detection of news.mpg: ", Histogram_Comparison_Gray(ngc_file_list))
    # print()