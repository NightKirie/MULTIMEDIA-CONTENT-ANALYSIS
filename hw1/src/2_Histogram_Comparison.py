def Histogram_Comparison(file_list):
    

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # news
    news_file_list = []
    for dir_path, _, file_list in os.walk("../news_out"):
        news_file_list = [os.path.join(dir_path, file) for file in file_list]
    print("Histogram comparison for shot-change detection of news.mpg: ", Histogram_Comparison(news_file_list))
    print()

    # soccer
    soccer_file_list = []
    for dir_path, _, file_list in os.walk("../soccer_out"):
        soccer_file_list = [os.path.join(dir_path, file) for file in file_list]
    print("Histogram comparison for shot-change detection of news.mpg: ", Histogram_Comparison(soccer_file_list))
    print()

    # ngc
    ngc_file_list = []
    for dir_path, _, file_list in os.walk("../ngc_out"):
        ngc_file_list = [os.path.join(dir_path, file) for file in file_list]
    print("Histogram comparison for shot-change detection of news.mpg: ", Histogram_Comparison(ngc_file_list))
    print()