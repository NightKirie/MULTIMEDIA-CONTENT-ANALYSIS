import cv2
import numpy as np
import os
import skiamge as sk

# """
#     bt: threshold for blue value difference
#     gt: threshold for green value difference
#     rt: threshold for red value difference
#     bcp: pixel change percentage for blue value difference over threshold
#     gcp: pixel change percentage for green value difference over threshold
#     rcp: pixel change percentage for red value difference over threshold
# """
# def Pair_Wise_Comparison_BGR(file_list, bt=20, gt=20, rt=20, bcp=0.2, gcp=0.2, rcp=0.2):
#     img_1 = cv2.imread(file_list[0]).astype("int16")
#     img_pixel = img_1.shape[0] * img_1.shape[1]

#     for i in range(1, len(file_list)-1):
#         img_2 = cv2.imread(file_list[i]).astype("int16")
#         if (abs(img_1[:,:,0] - img_2[:,:,0]) > bt).sum() > (img_pixel * bcp) and\
#            (abs(img_1[:,:,0] - img_2[:,:,0]) > gt).sum() > (img_pixel * gcp) and\
#            (abs(img_1[:,:,0] - img_2[:,:,0]) > rt).sum() > (img_pixel * rcp):
#             print(i)
#         img_1 = img_2


"""
    t: threshold for gray value difference
    cp: pixel change percentage for gray value difference over threshold
"""
def Pair_Wise_Comparison(file_list, t=15, cp=0.4):
    img_1 = cv2.cvtColor(cv2.imread(file_list[0]), cv2.COLOR_BGR2GRAY).astype("int16")
    img_pixel = img_1.shape[0] * img_1.shape[1]
    shot_change_index = []

    for i in range(1, len(file_list)-1):
        img_2 = cv2.cvtColor(cv2.imread(file_list[i]), cv2.COLOR_BGR2GRAY).astype("int16")
        if (abs(img_1 - img_2) > t).sum() > (img_pixel * cp):
            shot_change_index.append(i)
        img_1 = img_2   

    return shot_change_index

"""
    t: threshold for gray value difference
    cp: pixel change percentage for gray value difference over threshold
    w: window size for smoothing comparison
"""
def Pair_Wise_Comparison_Window(file_list, t=15, cp=0.4, w=3):
    img_1 = cv2.cvtColor(cv2.imread(file_list[0]), cv2.COLOR_BGR2GRAY).astype("int16")
    img_pixel = img_1.shape[0] * img_1.shape[1]
    shot_change_index = []

    for i in range(1, len(file_list)-1):
        img_2 = cv2.cvtColor(cv2.imread(file_list[i]), cv2.COLOR_BGR2GRAY).astype("int16")
        if (abs(img_1 - img_2) > t).sum() > (img_pixel * cp):
            shot_change_index.append(i)
        img_1 = img_2   

    return shot_change_index

# """
#     t: threshold for gray value difference
#     cp: pixel change percentage for gray value difference over threshold
#     dl: dissolve length(frames) in video
# """
# def Pair_Wise_Comparison_Dissolve(file_list, t=20, cp=0.2, dl=5):
    
#     img_list = [cv2.cvtColor(cv2.imread(file_list[0]), cv2.COLOR_BGR2GRAY).astype("int16")]
#     img_pixel = img_list[0].shape[0] * img_list[0].shape[1]
#     for i in range(1, dl):
#         img_list = np.concatenate((img_list, [cv2.cvtColor(cv2.imread(file_list[i]), cv2.COLOR_BGR2GRAY).astype("int16")]))
#     shot_change_index = []
    
#     for i in range(dl, len(file_list)-1):
#         is_dissolving = True
#         for j in range(0, dl-1):
#             is_dissolving = is_dissolving and ((abs(img_list[j] - img_list[j+1]) > t).sum() > (img_pixel * cp))
        
#         if is_dissolving:
#             shot_change_index.append(f"{i-dl}~{i}")
#         img_list = img_list[1:, :, :]
#         img_list = np.concatenate((img_list, [cv2.cvtColor(cv2.imread(file_list[i]), cv2.COLOR_BGR2GRAY).astype("int16")]))
        
#         # img_2 = cv2.cvtColor(cv2.imread(file_list[i]), cv2.COLOR_BGR2GRAY).astype("int16")
#         # if (abs(img_1 - img_2) > t).sum() > (img_pixel * cp):
#         #     shot_change_index.append(i)
#         # img_1 = img_2   

#     return shot_change_index

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # news
    news_file_list = []
    for dir_path, _, file_list in os.walk("../news_out"):
        news_file_list = [os.path.join(dir_path, file) for file in file_list]
    print("Default pair-wise comparison for shot-change detection of news.mpg: ", Pair_Wise_Comparison(news_file_list))
    print("Windowed pair-wise comparison for shot-change detection of news.mpg: ", Pair_Wise_Comparison_Window(news_file_list))

    # soccer
    soccer_file_list = []
    for dir_path, _, file_list in os.walk("../soccer_out"):
        soccer_file_list = [os.path.join(dir_path, file) for file in file_list]
    print("Default pair-wise comparison for shot-change detection of soccer.mpg: ", Pair_Wise_Comparison(soccer_file_list))
    print("Windowed pair-wise comparison for shot-change detection of soccer.mpg: ", Pair_Wise_Comparison_Window(soccer_file_list))

    # ngc
    ngc_file_list = []
    for dir_path, _, file_list in os.walk("../ngc_out"):
        ngc_file_list = [os.path.join(dir_path, file) for file in file_list]
    print("Default pair-wise comparison for shot-change detection of ngc.mpg: ", Pair_Wise_Comparison(ngc_file_list))
    print("Windowed pair-wise comparison for shot-change detection of ngc.mpg: ", Pair_Wise_Comparison_Window(ngc_file_list))