import matplotlib.pyplot as plt
import torch

from sklearn.manifold import TSNE

LABEL_TO_COLOR = {x: c for x, c in enumerate(['#a11a25', '#feadbb', '#3145ee', '#9aaaff', '#e8fd00',
                                              '#51ffbc', '#fe5c0d', '#fe00b9', '#77662c', '#00810a'])}


def vis(model, dataloader, epoch: int, limit=2500, device=torch.device('cpu')):
    if dataloader.dataset.tensors[0].shape[0] > limit:
        perm = torch.randperm(dataloader.dataset.tensors[0].size(0))
        idx = perm[:2500]
        samples = dataloader.dataset.tensors[0][idx]
        true_labels = dataloader.dataset.tensors[1][idx]
    else:
        samples = dataloader.dataset.tensors[0]
        true_labels = dataloader.dataset.tensors[1]

    x_latent = model.encoder(samples.to(device)).detach().cpu().numpy()

    if x_latent.shape[1] > 2:
        spaces = ['latent (tSNE)', '']
        latent = TSNE(n_components=2,
                      learning_rate='auto',
                      init='random',
                      perplexity=min([100, int(samples.shape[0] / 100)])).fit_transform(x_latent)
    else:
        spaces = ['latent', '']
        latent = x_latent

    if samples.shape[1] > 2:
        spaces[1] = spaces[0]
        spaces_for_prediction = latent
    else:
        spaces[1] = 'original'
        spaces_for_prediction = samples

    _, _, fig_labels = model(samples.to(device))

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    if latent.reshape((samples.shape[0], -1)).shape[1] == 1:
        ax[0].scatter(latent.reshape(-1),
                      np.zeros(latent.reshape(-1).shape[0]),
                      color=[LABEL_TO_COLOR[i] for i in true_labels.detach().cpu().numpy()])
        ax[0].set_title(f'True Partition in {spaces[0]}')
    else:
        ax[0].scatter(latent[:, 0],
                      latent[:, 1],
                      color=[LABEL_TO_COLOR[i] for i in true_labels.detach().cpu().numpy()])
        ax[0].set_title(f'True Partition in {spaces[0]}')

    ax[1].scatter(spaces_for_prediction[:, 0],
                  spaces_for_prediction[:, 1],
                  color=[LABEL_TO_COLOR[i] for i in fig_labels.detach().cpu().numpy().argmax(axis=1).astype(int)])
    ax[1].set_title(f'Predicted Partition in {spaces[1]}')

    plt.show()
