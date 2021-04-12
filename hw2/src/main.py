import os
import numpy as np
from collections import Counter
from skimage import io
from skimage.color import rgb2hsv, rgb2gray
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

MIN_COMPONENT = 2
MAX_COMPONENT = 21

def q1_rgb():
    img1 = io.imread("../data/soccer1.jpg")
    img1_height = img1.shape[0]
    img1_width = img1.shape[1]
    img1 = img1.reshape(-1, 3)
    
    with open("../result/rgb/q1/acc.txt", "w+") as f:
        acc_list = []
        for i in range(MIN_COMPONENT, MAX_COMPONENT):
            gmm = GaussianMixture(n_components=i, covariance_type="full")
            gmm.fit(img1)
            img1_predict = gmm.predict(img1)
            img1_predict = Seperate_Playfield(img1_predict, i)
            img1_mask = rgb2gray(io.imread("../data/soccer1_mask.png")).reshape(-1)
            acc = Calculate_Accuracy(img1_predict, img1_mask)
            f.write(f"Accuracy of gaussian Mixture with {i} component: {str(acc)}\n")
            acc_list.append(acc)
            img1_predict = img1_predict.reshape(img1_height, img1_width)
            io.imsave(f"../result/rgb/q1/img1_{i}.jpg", img1_predict)
            
        f.write("\n")    
        f.write(f"Average of accuracy is {str(np.mean(acc_list))}")
        
        fig, axes = plt.subplots(1,1)
        axes.plot([x for x in range(MIN_COMPONENT, MAX_COMPONENT)], acc_list)
        axes.xaxis.set_major_locator(MaxNLocator(20)) 
        plt.savefig("../result/rgb/q1/acc.jpg")
        plt.clf()

def q1_hsv():
    img1 = rgb2hsv(io.imread("../data/soccer1.jpg"))
    img1_height = img1.shape[0]
    img1_width = img1.shape[1]
    img1 = img1.reshape(-1, 3)
    
    with open("../result/hsv/q1/acc.txt", "w+") as f:
        acc_list = []
        for i in range(MIN_COMPONENT, MAX_COMPONENT):
            gmm = GaussianMixture(n_components=i, covariance_type="full")
            gmm.fit(img1)
            img1_predict = gmm.predict(img1)
            img1_predict = Seperate_Playfield(img1_predict, i)
            img1_mask = rgb2gray(io.imread("../data/soccer1_mask.png")).reshape(-1)
            acc = Calculate_Accuracy(img1_predict, img1_mask)
            f.write(f"Accuracy of gaussian Mixture with {i} component: {str(acc)}\n")
            acc_list.append(acc)
            img1_predict = img1_predict.reshape(img1_height, img1_width)
            io.imsave(f"../result/hsv/q1/img1_{i}.jpg", img1_predict)
        
        f.write("\n")    
        f.write(f"Average of accuracy is {str(np.mean(acc_list))}")    
        
        fig, axes = plt.subplots(1,1)
        axes.plot([x for x in range(MIN_COMPONENT, MAX_COMPONENT)], acc_list)
        axes.xaxis.set_major_locator(MaxNLocator(20)) 
        plt.savefig("../result/hsv/q1/acc.jpg")
        plt.clf()

def q2_rgb():
    img1 = io.imread("../data/soccer1.jpg")
    img1_height = img1.shape[0]
    img1_width = img1.shape[1]
    img1 = img1.reshape(-1, 3)

    img2 = io.imread("../data/soccer2.jpg")
    img2_height = img2.shape[0]
    img2_width = img2.shape[1]
    img2 = img2.reshape(-1, 3)
    
    with open("../result/rgb/q2/acc.txt", "w+") as f:
        acc_list = []
        for i in range(MIN_COMPONENT, MAX_COMPONENT):
            gmm = GaussianMixture(n_components=i, covariance_type="full")
            gmm.fit(img1)
            img2_predict = gmm.predict(img2)
            img2_predict = Seperate_Playfield(img2_predict, i)
            img2_mask = rgb2gray(io.imread("../data/soccer2_mask.png")).reshape(-1)
            acc = Calculate_Accuracy(img2_predict, img2_mask)
            f.write(f"Accuracy of gaussian Mixture with {i} component: {str(acc)}\n")
            acc_list.append(acc)
            img2_predict = img2_predict.reshape(img2_height, img2_width)
            io.imsave(f"../result/rgb/q2/img2_{i}.jpg", img2_predict)
        
        f.write("\n")    
        f.write(f"Average of accuracy is {str(np.mean(acc_list))}")

        fig, axes = plt.subplots(1,1)
        axes.plot([x for x in range(MIN_COMPONENT, MAX_COMPONENT)], acc_list)
        axes.xaxis.set_major_locator(MaxNLocator(20)) 
        plt.savefig("../result/rgb/q2/acc.jpg")
        plt.clf()

def q2_hsv():
    img1 = rgb2hsv(io.imread("../data/soccer1.jpg"))
    img1_height = img1.shape[0]
    img1_width = img1.shape[1]
    img1 = img1.reshape(-1, 3)

    img2 = rgb2hsv(io.imread("../data/soccer2.jpg"))
    img2_height = img2.shape[0]
    img2_width = img2.shape[1]
    img2 = img2.reshape(-1, 3)
    
    with open("../result/hsv/q2/acc.txt", "w+") as f:
        acc_list = []
        for i in range(MIN_COMPONENT, MAX_COMPONENT):
            gmm = GaussianMixture(n_components=i, covariance_type="full")
            gmm.fit(img1)
            img2_predict = gmm.predict(img2)
            img2_predict = Seperate_Playfield(img2_predict, i)
            img2_mask = rgb2gray(io.imread("../data/soccer2_mask.png")).reshape(-1)
            acc = Calculate_Accuracy(img2_predict, img2_mask)
            f.write(f"Accuracy of gaussian Mixture with {i} component: {str(acc)}\n")
            acc_list.append(acc)
            img2_predict = img2_predict.reshape(img2_height, img2_width)
            io.imsave(f"../result/hsv/q2/img2_{i}.jpg", img2_predict)
            
        f.write("\n")    
        f.write(f"Average of accuracy is {str(np.mean(acc_list))}")
        
        fig, axes = plt.subplots(1,1)
        axes.plot([x for x in range(MIN_COMPONENT, MAX_COMPONENT)], acc_list)
        axes.xaxis.set_major_locator(MaxNLocator(20)) 
        plt.savefig("../result/hsv/q2/acc.jpg")
        plt.clf()

def q3_rgb():
    img1 = io.imread("../data/soccer1.jpg")
    img1_height = img1.shape[0]
    img1_width = img1.shape[1]
    img1 = img1.reshape(-1, 3)
    
    img2 = io.imread("../data/soccer2.jpg")
    img2_height = img2.shape[0]
    img2_width = img2.shape[1]
    img2 = img2.reshape(-1, 3)
    
    with open("../result/rgb/q3/acc.txt", "w+") as f:
        acc_list_1 = []
        acc_list_2 = []
        for i in range(MIN_COMPONENT, MAX_COMPONENT):
            gmm = GaussianMixture(n_components=i, covariance_type="full")
            gmm.fit(np.concatenate((img1, img2)))

            img1_predict = gmm.predict(img1)
            img1_predict = Seperate_Playfield(img1_predict, i)
            img1_mask = rgb2gray(io.imread("../data/soccer1_mask.png")).reshape(-1)
            acc = Calculate_Accuracy(img1_predict, img1_mask)
            f.write(f"Accuracy of image 1 of gaussian Mixture with {i} component: {str(acc)}\n")
            acc_list_1.append(acc)
            img1_predict = img1_predict.reshape(img1_height, img1_width)
            io.imsave(f"../result/rgb/q3/img1_{i}.jpg", img1_predict)

            img2_predict = gmm.predict(img2)
            img2_predict = Seperate_Playfield(img2_predict, i)
            img2_mask = rgb2gray(io.imread("../data/soccer2_mask.png")).reshape(-1)
            acc = Calculate_Accuracy(img2_predict, img2_mask)
            f.write(f"Accuracy of image 2 of gaussian Mixture with {i} component: {str(acc)}\n")
            acc_list_2.append(acc)
            img2_predict = img2_predict.reshape(img2_height, img2_width)
            io.imsave(f"../result/rgb/q3/img2_{i}.jpg", img2_predict)
            
        f.write("\n")    
        f.write(f"Average of accuracy image 1 is {str(np.mean(acc_list_1))}")

        f.write("\n")    
        f.write(f"Average of accuracy image 2 is {str(np.mean(acc_list_2))}")
        
        fig, axes = plt.subplots(1,1)
        axes.plot([x for x in range(MIN_COMPONENT, MAX_COMPONENT)], acc_list_1)
        axes.xaxis.set_major_locator(MaxNLocator(20)) 
        plt.savefig("../result/rgb/q3/acc_1.jpg")
        plt.clf()

        fig, axes = plt.subplots(1,1)
        axes.plot([x for x in range(MIN_COMPONENT, MAX_COMPONENT)], acc_list_2)
        axes.xaxis.set_major_locator(MaxNLocator(20)) 
        plt.savefig("../result/rgb/q3/acc_2.jpg")
        plt.clf()


def q3_hsv():
    img1 = rgb2hsv(io.imread("../data/soccer1.jpg"))
    img1_height = img1.shape[0]
    img1_width = img1.shape[1]
    img1 = img1.reshape(-1, 3)
    
    img2 = rgb2hsv(io.imread("../data/soccer2.jpg"))
    img2_height = img2.shape[0]
    img2_width = img2.shape[1]
    img2 = img2.reshape(-1, 3)
    
    with open("../result/hsv/q3/acc.txt", "w+") as f:
        acc_list_1 = []
        acc_list_2 = []
        for i in range(MIN_COMPONENT, MAX_COMPONENT):
            gmm = GaussianMixture(n_components=i, covariance_type="full")
            gmm.fit(np.concatenate((img1, img2)))

            img1_predict = gmm.predict(img1)
            img1_predict = Seperate_Playfield(img1_predict, i)
            img1_mask = rgb2gray(io.imread("../data/soccer1_mask.png")).reshape(-1)
            acc = Calculate_Accuracy(img1_predict, img1_mask)
            f.write(f"Accuracy of image 1 of gaussian Mixture with {i} component: {str(acc)}\n")
            acc_list_1.append(acc)
            img1_predict = img1_predict.reshape(img1_height, img1_width)
            io.imsave(f"../result/hsv/q3/img1_{i}.jpg", img1_predict)

            img2_predict = gmm.predict(img2)
            img2_predict = Seperate_Playfield(img2_predict, i)
            img2_mask = rgb2gray(io.imread("../data/soccer2_mask.png")).reshape(-1)
            acc = Calculate_Accuracy(img2_predict, img2_mask)
            f.write(f"Accuracy of image 1 of gaussian Mixture with {i} component: {str(acc)}\n")
            acc_list_2.append(acc)
            img2_predict = img2_predict.reshape(img2_height, img2_width)
            io.imsave(f"../result/hsv/q3/img2_{i}.jpg", img2_predict)
            
        f.write("\n")    
        f.write(f"Average of accuracy image 1 is {str(np.mean(acc_list_1))}")

        f.write("\n")    
        f.write(f"Average of accuracy image 2 is {str(np.mean(acc_list_2))}")
        
        fig, axes = plt.subplots(1,1)
        axes.plot([x for x in range(MIN_COMPONENT, MAX_COMPONENT)], acc_list_1)
        axes.xaxis.set_major_locator(MaxNLocator(20)) 
        plt.savefig("../result/hsv/q3/acc_1.jpg")
        plt.clf()

        fig, axes = plt.subplots(1,1)
        axes.plot([x for x in range(MIN_COMPONENT, MAX_COMPONENT)], acc_list_2)
        axes.xaxis.set_major_locator(MaxNLocator(20)) 
        plt.savefig("../result/hsv/q3/acc_2.jpg")
        plt.clf()


def Calculate_Accuracy(img, img_mask):
    img_mask[img_mask == 0] = 0
    img_mask[img_mask == 255] = 1
    tp = np.sum(img == img_mask)
    return tp / len(img)


def Seperate_Playfield(img, bi_num):
    bi_img = np.zeros(img.shape[0])
    for i in range(0, bi_num//2):
        index, number = Counter(img.ravel()).most_common(bi_num//2)[i]
        np.put(bi_img, np.where(img == index), 1)
    return bi_img
        
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # q1
    q1_rgb()
    q1_hsv()

    # q2
    q2_rgb()
    q2_hsv()

    # q3
    q3_rgb()
    q3_hsv()