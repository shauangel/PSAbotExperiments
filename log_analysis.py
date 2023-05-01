import re
import matplotlib.pyplot as plt
import config
import numpy as np


# log-likelihood
def lda_train_perplexity(filename):
    f_n = re.compile("c(\d+)\.log")
    chunksize = f_n.findall(filename)[0]
    p = re.compile("(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity")
    matches = [p.findall(l) for l in open(filename)]
    matches = [m for m in matches if len(m) > 0]
    tuples = [t[0] for t in matches]
    # perplexity = [float(t[1]) for t in tuples]
    likelihood = [float(t[0]) for t in tuples]
    updates = list(range(0, len(tuples) * int(chunksize), int(chunksize)))
    return likelihood, updates


# normalized perplexity
def epoch_norm_perplexity(filename):
    # 02:47:56 INFO: Epoch 9: Perplexity estimate: 42.44834830507931
    # 02:47:56 INFO: Epoch 9: Coherence estimate: -0.724959300278274
    p = re.compile("Perplexity estimate: (\d+\.\d+)")
    matches = [p.findall(l) for l in open(filename)]
    matches = [m for m in matches if len(m) > 0]
    tuples = [t[0] for t in matches]
    perplexity = [float(t) for t in tuples]
    epochs = list(range(1, len(tuples)+1))
    return perplexity, epochs


# normalized coherence
def epoch_umass(filename):
    p = re.compile("Coherence estimate: (-*\d+\.\d+)")
    matches = [p.findall(l) for l in open(filename)]
    matches = [m for m in matches if len(m) > 0]
    tuples = [t[0] for t in matches]
    coherence = [float(t) for t in tuples]
    epochs = list(range(1, len(tuples) + 1))
    return coherence, epochs


# analyze eta parameter experiment logs
def eta_experiment(filename):
    p = re.compile("(-*\d+\.\d+) per-word")
    matches = [p.findall(l) for l in open(filename)]
    matches = [m for m in matches if len(m) > 0]
    tuples = [t[0] for t in matches]
    likelihood = [float(t) for t in tuples]
    # print(likelihood)
    updates = list(range(0, len(tuples) * config.CHUNKSIZE, config.CHUNKSIZE))
    return likelihood, updates


# display final training result
def final_record(filename, last_epoch):
    pplx = re.compile("Epoch " + str(last_epoch) + ": Perplexity estimate: (\d+\.\d+)")
    ch = re.compile("Epoch " + str(last_epoch) + ": Coherence estimate: (-*\d+\.\d+)")
    time = re.compile("<<Training Time>>: (\d+\.\d+)")
    matches = [pplx.findall(l) + ch.findall(l) + time.findall(l) for l in open(filename)]
    matches = [m for m in matches if len(m) > 0]
    result = {'perplexity':matches[0][0], 'coherence':matches[1][0], 'time':matches[2][0]}
    print(result)
    return result


if __name__ == "__main__":
    # chunk size
    """
    log_files = ["c" + str(i) + ".log" for i in range(1, 11)]
    f_n = re.compile("c(\d+)\.log")
    for idx in range(len(log_files)):
        chunksize = f_n.findall(log_files[idx])[0]
        print(chunksize)
        # likelihood, updates = lda_train_perplexity('logs/chunksize/' + log_files[idx])
        p, epch = epoch_umass('logs/chunksize/' + log_files[idx])
        plt.plot(epch, p, linewidth=1, c=config.COLOR_LIST[idx],
                 label=chunksize, marker=config.MARKER_STYLE_LIST[idx])
        final_record('logs/chunksize/' + log_files[idx], 9)
    plt.ylabel("coherence")
    plt.xlabel("passes")
    plt.xlim(xmax=10)
    plt.title("Topic Model Coherence of Different Chunksize")
    plt.legend(loc="center right")
    plt.grid()
    plt.savefig("chunksize_coherence.png")
    plt.close()
    """

    # eta & alpha
    """
    params = list(np.logspace(-2, 1, 4, base=10))
    params = [str(i) if i < 1 else str(int(i)) for i in list(params)]
    for p in params:
        log_file = "logs/pairs" + p + ".log"
        # likelihood, updates = eta_experiment("logs/" + log_file)
        perplexity, epoch = epoch_norm_perplexity(log_file)
        # coherence, epoch = epoch_umass(log_file)
        plt.plot(epoch, perplexity, linewidth=1, c=config.COLOR_LIST[params.index(p)],
                 label=p, marker=config.MARKER_STYLE_LIST[params.index(p)])
    plt.ylabel("Perplexity")
    plt.xlabel("passes")
    plt.xlim(xmin=0, xmax=100)
    plt.title("Topic Model Perplexity")
    plt.legend(loc="center right")
    plt.grid()
    plt.savefig("pairs_100epochs_perplexity.png")
    plt.close()
    """

    # topic num
    params = list(range(2,10))

    for p in params:
        log_file = "topicnum" + str(p) + ".log"
        likelihood, updates = eta_experiment("logs/" + log_file)
        # perplexity, epoch = epoch_norm_perplexity(log_file)
        # coherence, epoch = epoch_umass(log_file)
        plt.plot(updates, likelihood, linewidth=1, c=config.COLOR_LIST[params.index(p)],
                 label=p, marker=config.MARKER_STYLE_LIST[params.index(p)])
        plt.ylabel("likelihood")
        plt.xlabel("iteration")
        plt.xlim(xmin=0, xmax=100)
        plt.title("Topic Model Convergence")
        plt.legend(loc="center right")
        plt.grid()
        plt.savefig("topicnum" + str(p) + ".png")
        plt.close()

    plt.clf()
    for p in params:
        log_file = "topicnum" + str(p) + ".log"
        likelihood, updates = eta_experiment("logs/" + log_file)
        #perplexity, epoch = epoch_norm_perplexity(log_file)
        #coherence, epoch = epoch_umass(log_file)
        plt.plot(updates, likelihood, linewidth=1, c=config.COLOR_LIST[params.index(p)],
                 label=p, marker=config.MARKER_STYLE_LIST[params.index(p)])
    plt.ylabel("likelihood")
    plt.xlabel("iteration")
    plt.xlim(xmin=0, xmax=100)
    plt.title("Topic Model Convergence")
    plt.legend(loc="center right")
    plt.grid()
    plt.savefig("topicnum_all.png")
    plt.close()

    # Perplexity
    plt.clf()
    for p in params:
        log_file = "topicnum" + str(p) + ".log"
        # likelihood, updates = eta_experiment("logs/" + log_file)
        perplexity, epoch = epoch_norm_perplexity("logs/" + log_file)
        #coherence, epoch = epoch_umass(log_file)
        plt.plot(epoch, perplexity, linewidth=1, c=config.COLOR_LIST[params.index(p)],
                 label=p, marker=config.MARKER_STYLE_LIST[params.index(p)])
    plt.ylabel("perplexity")
    plt.xlabel("epoch")
    plt.xlim(xmin=0, xmax=10)
    plt.title("Topic Model Perplexity")
    plt.legend(loc="center right")
    plt.grid()
    plt.savefig("topicnum_perplexity.png")
    plt.close()

    # Coherence
    # Perplexity
    plt.clf()
    for p in params:
        print(p)
        log_file = "topicnum" + str(p) + ".log"
        # likelihood, updates = eta_experiment("logs/" + log_file)
        # perplexity, epoch = epoch_norm_perplexity(log_file)
        coherence, epoch = epoch_umass("logs/" + log_file)
        plt.plot(epoch, coherence, linewidth=1, c=config.COLOR_LIST[params.index(p)],
                 label=p, marker=config.MARKER_STYLE_LIST[params.index(p)])
        final_record("logs/" + log_file, config.PASSES-1)
    plt.ylabel("coherence")
    plt.xlabel("epoch")
    plt.xlim(xmin=0, xmax=10)
    plt.title("Topic Model Coherence")
    plt.legend(loc="center right")
    plt.grid()
    plt.savefig("topicnum_coherence.png")
    plt.close()


