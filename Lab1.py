def accuracy_score(y_true,y_pred):
    x=0
    accuracy=0
    while x < len(y_true):
        accuracy += y_true[x]==y_pred[x]
        x+=1
    print(accuracy)
    accuracy /= len(y_true)
    print(accuracy)

def precision_recall_score(y_true, y_pred):
    tp=0
    fp=0
    fn=0
    x=0
    while x < len(y_pred):
        if y_pred[x]==1:
            if y_true[x]==1:
                tp+=1
                x+=1
            else:
                fp+=1
                x+=1
        else:
            if y_true[x]==1:
                fn+=1
                x+=1
            else:
                x+=1
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    return precision

y_pred = [1, 1, 1, 0, 1, 0, 1, 1, 0, 0]
y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]


def main():

    precision = precision_recall_score(y_true,y_pred)
    print(precision)
if __name__ == "__main__":
    main()
