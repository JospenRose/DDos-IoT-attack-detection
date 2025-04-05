import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from save_load import load


def classfi_report(y_test, predicted, k):

    # Classification report
    class_report = classification_report(y_test, predicted, output_dict=True)
    report_df = pd.DataFrame(class_report).transpose()

    # Plot the DataFrame
    plt.figure(figsize=(8, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues', fmt='.2f')

    plt.xticks(rotation=0, fontweight='bold', fontname='Serif')
    plt.yticks(fontweight='bold', fontname='Serif')
    plt.title('Classification Report', fontweight='bold', fontname='Serif')

    plt.savefig(f'Results/Classification Report learning rate - {k}.png')
    plt.show()


def bar_plot(label, data1, data2, metric):

    # create data
    df = pd.DataFrame([data1, data2],
                      columns=label)
    df1 = pd.DataFrame()
    df1['Learn Rate(%)'] = [70, 80]
    df = pd.concat((df1, df), axis=1)
    # plot grouped bar chart
    df.plot(x='Learn Rate(%)',
            kind='bar',
            stacked=False)

    plt.ylabel(metric, fontweight='bold', fontname='Serif')
    plt.xticks(rotation=0, fontweight='bold', fontname='Serif')
    plt.yticks(fontweight='bold', fontname='Serif')
    plt.xlabel('Learn Rate %', fontweight='bold', fontname='Serif')
    plt.legend(loc='center', prop={'weight': 'bold', 'family': 'Serif', 'size': 8})
    plt.title(metric, fontweight='bold', fontname='Serif')

    plt.savefig('./Results/'+metric+'.png', dpi=400)
    plt.show(block=False)


def densityplot(actual, predicted, learning_rate):

    plt.figure(figsize=(8, 6))
    sns.kdeplot(actual, color='orange', label='Actual',  fill=True)
    sns.kdeplot(predicted, color='blue', label='Predicted',  fill=True)

    plt.ylabel('Density', fontweight='bold', fontname='Serif')
    plt.xticks(rotation=0, fontweight='bold', fontname='Serif')
    plt.yticks(fontweight='bold', fontname='Serif')
    plt.xlabel('Value', fontweight='bold', fontname='Serif')
    plt.legend(loc='upper right', prop={'weight': 'bold', 'family': 'Serif'})
    plt.title("Actual vs Predicted values", fontweight='bold', fontname='Serif')

    plt.savefig(f'Results/Density Plot Learning rate-{learning_rate}.png')
    plt.show()


def confu_plot(y_test, y_pred, k):
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Purples, values_format='.0f', ax=ax)

    plt.ylabel('True Labels', fontweight='bold', fontname='Serif')
    plt.xticks(rotation=0, fontweight='bold', fontname='Serif')
    plt.yticks(fontweight='bold', fontname='Serif')
    plt.xlabel('Predicted Labels', fontweight='bold', fontname='Serif')
    plt.title('Confusion Matrix', fontweight='bold', fontname='Serif')
    plt.tight_layout()
    plt.savefig(f'Results/Confusion Matrix Learning rate-{k}.png')
    plt.show()


def plotres():

    # learning rate -  70  and 30

    alexnet_70 = load('alexnet_70')
    dnn_70 = load('dnn_70')
    cnn_70 = load('cnn_70')
    lstm_70 = load('lstm_70')
    proposed_70 = load('proposed_70')

    data = {
        'AlexNet': alexnet_70,
        'DNN': dnn_70,
        'CNN': cnn_70,
        'LSTM': lstm_70,
        'TriGuard-Net': proposed_70
    }

    ind = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'R2 Score', 'MCC', 'Cohen’s Kappa', 'Hamming Loss', 'Jaccard Score']
    table = pd.DataFrame(data, index=ind)
    print('---------- Metrics for 70 training 30 testing ----------')
    print(table)

    table.to_excel('./Results/table_70.xlsx')

    val1 = np.array(table)

    # learning rate -  80  and 20

    alexnet_80 = load('alexnet_80')
    dnn_80 = load('dnn_80')
    cnn_80 = load('cnn_80')
    lstm_80 = load('lstm_80')
    proposed_80 = load('proposed_80')

    data1 = {
        'AlexNet': alexnet_80,
        'DNN': dnn_80,
        'CNN': cnn_80,
        'LSTM': lstm_80,
        'TriGuard-Net': proposed_80
    }

    ind =['Accuracy', 'Precision', 'Recall', 'F1 Score', 'R2 Score', 'MCC', 'Cohen’s Kappa', 'Hamming Loss', 'Jaccard Score']
    table1 = pd.DataFrame(data1, index=ind)
    print('---------- Metrics for 80 training 20 testing ----------')
    print(table1)

    val2 = np.array(table1)
    table1.to_excel('./Results/table_80.xlsx')

    metrices = [val1, val2]

    mthod = ['AlexNet', 'DNN', 'CNN', 'LSTM', 'TriGuard-Net']
    metrices_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'R2 Score', 'MCC', 'Cohen’s Kappa', 'Hamming Loss', 'Jaccard Score']

    # Bar plot
    for i in range(len(metrices_plot)):
        bar_plot(mthod, metrices[0][i, :], metrices[1][i, :], metrices_plot[i])

    learn_data = [70, 80]
    for k in learn_data:
        y_test = load(f'y_test_{k}')
        y_pred = load(f'predicted_{k}')
        densityplot(y_test, y_pred, k)
        classfi_report(y_test, y_pred, k)

        confu_plot(y_test, y_pred, k)