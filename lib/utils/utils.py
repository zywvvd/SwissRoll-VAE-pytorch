import matplotlib.pyplot as plt


def show(samples, name='Samples'):
    Xs, Ys = samples[:,0], samples[:,1]
    plt.figure(123, figsize=(8,8))

    plt.clf()
    plt.scatter(Xs, Ys)
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.title(name)
    plt.pause(.002)
    