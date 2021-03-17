import cv2
import os

def Pixel_Difference(file_list, t=10):
    img_1 = cv2.imread(file_list[0])
    cv2.imshow("img_1", img_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #for i in (0, len(file_list)-2):



if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    soccer_file_list = []
    for dir_path, _, file_list in os.walk("../soccer_out"):
        soccer_file_list = [os.path.join(dir_path, file) for file in file_list]

    # print(soccer_file_list)
    Pixel_Difference(soccer_file_list)