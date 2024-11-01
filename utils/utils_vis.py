import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

from scipy.spatial import distance


def prepare_data(ori_sig, gen_sig):
    # Analysis sample size (for faster computation)
    sample_num = min([1000, len(ori_sig)])
    idx = np.random.permutation(len(ori_sig))[:sample_num]

    # Data preprocessing
    # ori_ssig = np.asarray(ori_sig)
    # generated_data = np.asarray(gen_sig)

    ori_sig = ori_sig[idx]
    gen_sig = gen_sig[idx]
    no, seq_len, dim = ori_sig.shape
    prep_ori = np.reshape(np.mean(ori_sig[0, :, :], 1), [1, seq_len])
    prep_gen = np.reshape(np.mean(gen_sig[0, :, :], 1), [1, seq_len])
    for i in range(1, sample_num):
        prep_ori = np.concatenate((prep_ori,
                                    np.reshape(np.mean(ori_sig[i, :, :], 1), [1, seq_len])))
        prep_gen = np.concatenate((prep_gen,
                                        np.reshape(np.mean(gen_sig[i, :, :], 1), [1, seq_len])))
    return prep_ori, prep_gen, sample_num


def PCA_plot(prep_ori, prep_gen, anal_sample_no, logger, args):
    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

    # PCA Analysis
    pca = PCA(n_components=2)
    pca.fit(prep_ori)
    pca_results = pca.transform(prep_ori)
    pca_hat_results = pca.transform(prep_gen)

    # Plotting
    # plt.ion()
    f, ax = plt.subplots(1)
    plt.scatter(pca_results[:, 0], pca_results[:, 1],
                c=colors[:anal_sample_no], alpha=0.2, label="Original")
    plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1],
                c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

    ax.legend()
    plt.title('PCA plot')
    plt.xlabel('x-pca')
    plt.ylabel('y_pca')
    plt.show()
    logger.log_fig(f'{args.dataset}', f)
    plt.close()


def TSNE_plot(prep_ori, prep_gen, anal_sample_no, logger, args):
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]
    prep_data_final = np.concatenate((prep_ori, prep_gen), axis=0)

    # TSNE anlaysis
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(prep_data_final)

    # Plotting
    f, ax = plt.subplots(1)

    plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                c=colors[:anal_sample_no], alpha=0.2, label="Original")
    plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

    ax.legend()

    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.show()
    logger.log_fig(f'{args.dataset}', f)

    plt.close()


def density_plot(prep_ori, prep_gen, logger, args):
    f, ax = plt.subplots(1)

    sns.distplot(prep_ori, hist=False, kde=True, label='Original')
    sns.distplot(prep_gen, hist=False, kde=True, kde_kws={'linestyle': '--'}, label='TS2I')
    # Plot formatting
    plt.legend()
    plt.xlabel('Data Value')
    plt.ylabel('Data Density Estimate')
    plt.rcParams['pdf.fonttype'] = 42
    plt.title(args.dataset)
    plt.show()
    logger.log_fig(f'{args.dataset}', f)
    plt.close()


def jensen_shannon_divergence(prep_ori, prep_gen,logger):
    """
    method to compute the Jenson-Shannon Divergence of two probability distributions
    """
    p_ = sns.histplot(prep_ori.flatten(), label='Original', stat='probability', bins=200).patches
    plt.close()
    q_ = sns.histplot(prep_gen.flatten(), label='TS2I', stat='probability', bins=200).patches
    plt.close()
    p = np.array([h.get_height() for h in p_])
    q = np.array([h.get_height() for h in q_])
    if p.shape[0] < q.shape[0]:
        q = q[:p.shape[0]]
    else:
        p = p[:q.shape[0]]
    logger.log('JSD', distance.jensenshannon(p, q))




