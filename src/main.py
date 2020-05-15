from constants import *
from src.logReg import run_classifier as log_classifier
from src.naiveBayes import run_classifier as nb_classifier
from src.knn import run_classifier as knn_classifier


def plot_stats(filename, stat_list, title, ylabel):
    classifiers = ["Log Reg", "Naive Bayes", "KNN"]
    plt.style.use("ggplot")
    plt.figure()
    plt.title(title)
    plt.ylabel(ylabel)

    plt.bar(classifiers, stat_list, color="royalblue", alpha=0.7)
    plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
    plt.savefig(filename)
    plt.close()


def main():
    acc_list = []
    time_list = []
    ## LOGISTIC REGRESSION ##
    print("**** Logistic Regression *****")
    acc, time = log_classifier()
    acc_list.append(acc)
    time_list.append(time)

    ## NAIVE BAYES ##
    print("***** Naive Bayes *****")
    acc, time = nb_classifier()
    acc_list.append(acc)
    time_list.append(time)

    ## KNN CLASSIFIER ##
    print("**** KNN Classfifier **** ")
    acc, time = knn_classifier()
    acc_list.append(acc)
    time_list.append(time)

    print("Accuracies:", acc_list)
    plot_stats('../output/acc_bar.png', acc_list, "Classification Accuracies", 'Accuracy')

    print("Prediction Times", time_list)
    plot_stats('../output/time_bar.png', time_list, "Prediction Times", "Runtime (Seconds)")


if __name__ == '__main__':
    main()
