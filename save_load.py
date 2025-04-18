import pickle


def save(name, value):
    with open('Saved Data/'+name+'.pkl', 'wb') as file:
        pickle.dump(value, file)


def load(name):
    with open('Saved Data/'+name+'.pkl', 'rb') as file:
        return pickle.load(file)
