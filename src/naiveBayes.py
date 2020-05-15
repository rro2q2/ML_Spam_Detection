from constants import *


def run_classifier():
    data = pd.read_csv(SPAM_DATA_DIR, encoding="latin")

    labels = []
    for item in data.v1:
        if item == "ham":
            labels.append(0)
        elif item == "spam":
            labels.append(1)

    labels = pd.Series(labels)

    # Setting X equal to emails and y equal to labels
    X = data.v2
    y = labels


    # Creating train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print("Data Lengths")
    print("\tTrain X:\t{}\n".format(len(X_train)),
          "\tTest X:\t{}\n".format(len(X_test)),
          "\tTrain y:\t{}\n".format(len(y_train)),
          "\tTest y:\t{} \n".format(len(y_test))
          )

    ham_ctr = 0
    spam_ctr = 0
    print("Number of Spam/Ham")
    for item in y:
        if item == 1:
            spam_ctr += 1
        elif item == 0:
            ham_ctr += 1

    print("\tHam:\t{}\n".format(ham_ctr),
          "\tSpam:\t{}\n".format(spam_ctr)
          )

    # Count vectorizer is used to count the number of times
    # each word comes up in each email
    vectorizer = CountVectorizer()
    counts = vectorizer.fit_transform(X_train.values)

    # Instantiating Multinomial Naive Bayes Classifier
    classifer = MultinomialNB()
    targets = y_train.values
    classifer.fit(counts, targets)

    example_count = vectorizer.transform(X_test.values)

    start = timeit.default_timer()
    predIdxs = classifer.predict(example_count)
    stop = timeit.default_timer()
    time = stop-start
    print("Overall training time: ", time)

    y_true = [item for item in predIdxs]
    y_pred = [item for item in y_test.values]

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # show a nicely formatted classification report
    print(classification_report(y_true, y_pred))

    # compute the confusion matrix and and use it to derive the raw
    # accuracy, sensitivity, and specificity
    cm = confusion_matrix(y_true, y_pred)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    conf_matrix = pd.crosstab(y_true, y_pred)
    print("Confusion Matrix:\n", conf_matrix)

    print("acc: {:.4f}".format(acc))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))

    # Plot confusion matrix
    plot_confusion_matrix('../output/naiveBayes_cm.png', cm, ['ham', 'spam'], title="Naive Bayes Confusion Matrix",
                          cmap=plt.cm.Blues, normalize=False)
    plt.close()
    return acc, time
