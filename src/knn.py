from constants import *


class KNN_NLC_Classifer():
    def __init__(self, k=1, distance_type='path'):
        self.k = k
        self.distance_type = distance_type

    # This function is used for training
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    # This function runs the K(1) nearest neighbour algorithm and
    # returns the label with closest match.
    def predict(self, x_test):
        self.x_test = x_test
        y_predict = []

        for item in x_test:
            max_sim = 0
            max_index = 0
            j = 0
            for sent in self.x_train:
                temp = self.document_similarity(item, sent)
                if temp > max_sim:
                    max_sim = temp
                    max_index = j
                j += 1
            y_predict.append(self.y_train[max_index])

        return y_predict

    def convert_tag(self, tag):
        """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
        tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
        try:
            return tag_dict[tag[0]]
        except KeyError:
            return None

    def doc_to_synsets(self, doc):
        """
            Returns a list of synsets in document.
            Tokenizes and tags the words in the document doc.
            Then finds the first synset for each word/tag combination.
        If a synset is not found for that combination it is skipped.

        Args:
            doc: string to be converted

        Returns:
            list of synsets
        """
        tokens = word_tokenize(doc + ' ')

        l = []
        tags = nltk.pos_tag([tokens[0] + ' ']) if len(tokens) == 1 else nltk.pos_tag(tokens)

        for token, tag in zip(tokens, tags):
            syntag = self.convert_tag(tag[1])
            syns = wn.synsets(token, syntag)
            if (len(syns) > 0):
                l.append(syns[0])
        return l

    def similarity_score(self, s1, s2, distance_type='path'):
        """
        Calculate the normalized similarity score of s1 onto s2
        For each synset in s1, finds the synset in s2 with the largest similarity value.
        Sum of all of the largest similarity values and normalize this value by dividing it by the
        number of largest similarity values found.

        Args:
            s1, s2: list of synsets from doc_to_synsets

        Returns:
            normalized similarity score of s1 onto s2
        """
        s1_largest_scores = []

        for i, s1_synset in enumerate(s1, 0):
            max_score = 0
            for s2_synset in s2:
                if distance_type == 'path':
                    score = s1_synset.path_similarity(s2_synset, simulate_root=False)
                else:
                    score = s1_synset.wup_similarity(s2_synset)
                if score != None:
                    if score > max_score:
                        max_score = score

            if max_score != 0:
                s1_largest_scores.append(max_score)

        mean_score = np.mean(s1_largest_scores)

        return mean_score

    def document_similarity(self, doc1, doc2):
        """Finds the symmetrical similarity between doc1 and doc2"""

        synsets1 = self.doc_to_synsets(doc1)
        #print(synsets1)
        synsets2 = self.doc_to_synsets(doc2)
        #print(synsets2)

        return (self.similarity_score(synsets1, synsets2) + self.similarity_score(synsets2, synsets1)) / 2


def run_classifier():

    dataset = pd.read_csv(SPAM_DATA_DIR, encoding='latin')[:1000]
    dataset.rename(columns = {'v1':'Spam/Not_Spam','v2':'message'})
    #dataset['output'] = np.where(dataset['message'] == 'spam', 1, 0)
    Num_Words = dataset.shape[0]

    # remove stop words
    stop_words = set(stopwords.words('english'))
    numbers = '1234567890'
    message = [item for item in dataset.v2]

    X = []
    # Remove stopwords
    for item in message:
        word_tokens = word_tokenize(item)
        filtered_sentence = [w for w in word_tokens if w not in stop_words]
        X.append(filtered_sentence)
    # Convert text to lower
    X = [[w.lower() for w in item] for item in X]

    # Remove numeric content
    for item in X:
        for word in item:
            diff = set(numbers) - set(word)
            if len(diff) < len(set(numbers)):
                item.remove(word)

    # Join words together
    inputs = []
    for item in X:
        temp = ""
        for w in item:
            temp += " " + w
        inputs.append(temp)

    X = []
    for item in inputs:
        temp = ""
        for c in item:
            if c not in punctuation:
                temp += ''.join(c)
        X.append(temp)

    X = pd.Series(X)
    test = X
    # Get labels
    labels = []
    for item in dataset.v1:
        if item == "ham":
            labels.append(0)
        elif item == "spam":
            labels.append(1)

    y = labels #pd.Series(labels)

    # Split data
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data Lengths")
    print("\tTrain X:\t{}\n".format(len(train_X)),
          "\tTest X:\t{}\n".format(len(test_X)),
          "\tTrain y:\t{}\n".format(len(train_y)),
          "\tTest y:\t{} \n".format(len(test_y))
          )

    ham_ctr = 0
    spam_ctr = 0
    print("Number of Spam/Ham")
    for item in y:
        if item == 1: spam_ctr += 1
        elif item == 0: ham_ctr += 1

    print("\tHam:\t{}\n".format(ham_ctr),
          "\tSpam:\t{}\n".format(spam_ctr)
          )

    # Fit data
    classifier = KNN_NLC_Classifer(k=1, distance_type='path')
    classifier.fit(train_X, train_y)

    start = timeit.default_timer()
    predIdxs = classifier.predict(test_X)
    stop = timeit.default_timer()
    time = stop-start
    print("Overall training time:", time)
    y_true = [item for item in predIdxs]
    y_pred = [item for item in test_y]

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
    plot_confusion_matrix('../output/knn_cm.png', cm, ['ham', 'spam'], title="KNN Confusion Matrix",
                          cmap=plt.cm.Blues, normalize=False)
    plt.close()
    return acc, time
