import re
import matplotlib.pyplot as plt
import config
import tensorflow as tf
import tensorflow_hub as hub


# normalized perplexity
def epoch_perplexity_list(filename, epoch):
    p = re.compile("Epoch " + str(epoch) + ": Perplexity estimate: (\d+\.\d+)")
    matches = [p.findall(l) for l in open(filename)]
    matches = [m for m in matches if len(m) > 0]
    tuples = [t[0] for t in matches]
    perplexity = [float(t) for t in tuples]
    return perplexity


# normalized coherence
def epoch_coherence_list(filename, epoch):
    p = re.compile("Epoch " + str(epoch) + ": Coherence estimate: (-*\d+\.\d+)")
    matches = [p.findall(l) for l in open(filename)]
    matches = [m for m in matches if len(m) > 0]
    tuples = [t[0] for t in matches]
    coherence = [float(t) for t in tuples]
    return coherence


def topic_sim(filename):
    embed = hub.KerasLayer("embeds/Wiki-words-250_2",
                           input_shape=[],
                           dtype=tf.string,
                           trainable=True,
                           name="Word_Embedding_Layer")

    word_pattern = r'\*"(.*?)"'
    topics = []
    similarities = []
    for l in open(filename):
        t = " ".join(re.findall(word_pattern, l))
        topics.append(t)
        if "topic #2" in l:
            vec = embed(topics)
            question_sim = []
            for sets in [(0, 1), (0, 2), (1, 2)]:
                question_sim.append(tf.losses.CosineSimilarity()(vec[sets[0]], vec[sets[1]]).numpy())
            print("avg_sim: " + str(sum(question_sim)/3))
            similarities.append(sum(question_sim)/3)
            topics.clear()
    return similarities


if __name__ == "__main__":
    posts = [3, 5, 7, 10]
    post_sim = []
    post_per = []
    post_coh = []
    for post_num in posts:
        file = "logs/posts/post_num" + str(post_num) + "_topic.log"
        sim = topic_sim(file)
        post_sim.append(abs(sum(sim)/len(sim)))

        all_perplexity = []
        all_coherence = []
        for i in range(10):
            all_perplexity.append(epoch_perplexity_list(file, i))
            all_coherence.append(epoch_coherence_list(file, i))

        post_per.append(sum(all_perplexity[9]) / 10)
        post_coh.append(sum(all_coherence[9]) / 10)
    # display similarity

    fig, ax = plt.subplots()
    bar_container = ax.bar(range(1,5), post_sim,
                           tick_label=posts,
                           fill=False,
                           hatch=config.BAR_FILL_STYLE[:len(posts)])
    ax.set(ylabel="post number",
           xlabel="average cosine similarity",
           ylim=(0.8, 1.0),
           title="Average Similarity of Different Document Amount")
    ax.bar_label(bar_container, fmt=lambda x: '{:.4f}'.format(x))
    plt.grid()
    plt.savefig("post_similarity.png")
    plt.close()


    """
    fig, ax = plt.subplots()
    bar_container = ax.bar(range(1, 5), post_per,
                           tick_label=posts,
                           fill=False,
                           hatch=config.BAR_FILL_STYLE[:len(posts)])
    ax.set(ylabel="post number",
           xlabel="average perplexity",
           ylim=(40, 55),
           title="Average Perplexity of Different Document Amount")
    ax.bar_label(bar_container, fmt=lambda x: '{:.3f}'.format(x))
    # line chart of each post num
    plt.grid()
    plt.savefig("post_perplexity.png")
    plt.close()
    """
    """
    for i in range(10):
        perplx = [p[i] for p in all_perplexity]
        epoch = range(10)
        plt.plot(epoch, perplx, linewidth=1, c=config.COLOR_LIST[i],
                label=i, marker=config.MARKER_STYLE_LIST[i])
    
    
    plt.ylabel("perplexity")
    plt.xlabel("passes")
    plt.xlim(xmax=10)
    plt.title("Topic Model Perplexity of x10 on 3 Posts")
    plt.legend(loc="center right")
    plt.grid()
    plt.savefig("7_posts_perplexity.png")
    plt.close()

    for i in range(10):
        coh = [p[i] for p in all_coherence]
        epoch = range(10)
        plt.plot(epoch, coh, linewidth=1, c=config.COLOR_LIST[i],
                label=i, marker=config.MARKER_STYLE_LIST[i])
    plt.ylabel("coherence")
    plt.xlabel("passes")
    plt.xlim(xmax=10)
    plt.title("Topic Model Coherence of x10 on 3 Posts")
    plt.legend(loc="center right")
    plt.grid()
    plt.savefig("7_posts_coherence.png")
    plt.close()
    """
