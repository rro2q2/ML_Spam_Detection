from constants import *


def text_preprocess(text):
    text = text.translate(str.maketrans('', '', punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)


def run_classifier():
    message_data = pd.read_csv(SPAM_DATA_DIR, encoding="latin")

    message_data = message_data.rename(columns = {'v1':'Spam/Not_Spam','v2':'message'})

    message_data_copy = message_data['message'].copy()

    message_data_copy = message_data_copy.apply(text_preprocess)

    vectorizer = TfidfVectorizer("english")

    message_mat = vectorizer.fit_transform(message_data_copy)

    message_data['length'] = message_data['message'].apply(len)

    length = message_data['length'].to_numpy()
    new_mat = np.hstack((message_mat.todense(), length[:, None]))

    message_train, message_test, spam_nospam_train, spam_nospam_test = train_test_split(new_mat,
                                                        message_data['Spam/Not_Spam'], test_size=0.3, random_state=20)

    Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
    Spam_model.fit(message_train, spam_nospam_train)

    start = timeit.default_timer()
    pred = Spam_model.predict(message_test)
    stop = timeit.default_timer()
    time = stop-start
    print("Overall training time:", time)

    y_true = [item for item in spam_nospam_test]
    y_pred = [item for item in pred]

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
    plot_confusion_matrix('../output/logReg_cm.png', cm, ['ham', 'spam'], title="Logistic Regression Confusion Matrix",
                          cmap=plt.cm.Blues, normalize=False)
    plt.close()
    return acc, time
