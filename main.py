from datagen import datagen
from save_load import load, save
from Detection import TriGuardNet, CNN, lstm, dnn, AlexNet
import matplotlib.pyplot as plt
from plot_result import plotres
from FHO import OriginalFHO, fit_func_70, fit_func_80

import os

os.makedirs('Saved Data', exist_ok=True)
os.makedirs('Data visualization', exist_ok=True)
os.makedirs('Results', exist_ok=True)


def main():
    datagen()

    # 70% training and 30% testing
    x_train_70 = load('x_train_70')
    y_train_70 = load('y_train_70')
    x_test_70 = load('x_test_70')
    y_test_70 = load('y_test_70')

    # 80% training and 20% testing

    x_train_80 = load('x_train_80')
    y_train_80 = load('y_train_80')
    x_test_80 = load('x_test_80')
    y_test_80 = load('y_test_80')

    training_data = [(x_train_70, y_train_70, x_test_70, y_test_70, fit_func_70), (x_train_80, y_train_80, x_test_80, y_test_80, fit_func_80)]

    i = 70

    for train_data in training_data:
        X_train, y_train, X_test, y_test, fit_func = train_data

        problem_dict1 = {
            "fit_func": fit_func,
            "lb": [40, 16, 0.00001],  # epochs, batch_size, learning-rate
            "ub": [100, 32, 0.001],
            "minmax": "min",
        }

        epoch = 20
        pop_size = 10
        model = OriginalFHO(epoch, pop_size)
        best_position, best_fitness = model.solve(problem_dict1)

        epochs = int(best_position[0])
        learning_rate = best_position[2]
        batch_size = int(best_position[1])

        y_pred, met, history = TriGuardNet(X_train, y_train, X_test, y_test, epochs, batch_size, learning_rate)

        save(f'proposed_{i}', met)
        save(f'predicted_{i}', y_pred)

        plt.figure(figsize=(10, 4))

        plt.subplot(121)
        plt.plot(history.history['accuracy'], label=['Train Accuracy'])
        plt.plot(history.history['val_accuracy'], label=['Validation Accuracy'])
        plt.title('Accuracy', fontweight='bold', fontname='Serif')
        plt.xlabel('Epoch', fontweight='bold', fontname='Serif')
        plt.ylabel('Accuracy', fontweight='bold', fontname='Serif')
        plt.xticks(rotation=0, fontweight='bold', fontname='Serif')
        plt.yticks(fontweight='bold', fontname='Serif')
        plt.legend(loc='lower right', prop={'weight': 'bold', 'family': 'Serif'})

        plt.subplot(122)
        plt.plot(history.history['loss'], label=['Train Loss'])
        plt.plot(history.history['val_loss'], label=['Validation Loss'])
        plt.title('Loss', fontweight='bold', fontname='Serif')
        plt.xlabel('Epoch', fontweight='bold', fontname='Serif')
        plt.ylabel('Loss', fontweight='bold', fontname='Serif')
        plt.xticks(rotation=0, fontweight='bold', fontname='Serif')
        plt.yticks(fontweight='bold', fontname='Serif')
        plt.legend(loc='upper right', prop={'weight': 'bold', 'family': 'Serif'})

        plt.tight_layout()
        plt.savefig(f'Results/Accuracy Loss Graph Learning rate {i}.png')
        plt.show()

        pred, met = CNN(X_train, y_train, X_test, y_test)
        save(f'cnn_{i}', met)

        pred, met = lstm(X_train, y_train, X_test, y_test)
        save(f'lstm_{i}', met)

        pred, met = dnn(X_train, y_train, X_test, y_test)
        save(f'dnn_{i}', met)

        pred, met = AlexNet(X_train, y_train, X_test, y_test)
        save(f'alexnet_{i}', met)

        i += 10


a = 0
if a == 0:
    main()

plotres()
plt.show()

