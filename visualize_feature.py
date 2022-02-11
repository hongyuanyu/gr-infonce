from test import *

##visualize
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
visualize_feature_pkl = '{}_{}set_{}visualize_tsne_feature.pkl'.format(config['dataset'], config['test_set'], os.path.split(config['ckp_prefix'])[-1])
save_dir = './visualize_feature'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print('visualize_feature_pkl: {}'.format(visualize_feature_pkl))
if os.path.exists(visualize_feature_pkl):
    with open(visualize_feature_pkl, 'rb') as f:
        X_tsne = pickle.load(f)
else:
    all_feature = data[0]
    all_feature = all_feature.view(all_feature.size(0), -1).data.cpu().numpy()
    X_tsne = TSNE(n_components=2,random_state=33).fit_transform(all_feature)
    with open(visualize_feature_pkl, 'wb') as f:
        pickle.dump(X_tsne, f)

feature, view, seq_type, label = data
view_list = sorted(list(set(view)))
seq_type_list = sorted(list(set(seq_type)))
label_list = sorted(list(set(label)))
tsne_feature = X_tsne

save_dir = '{}_{}visualize_tsne_images'.format(config['model_name'], os.path.split(config['ckp_prefix'])[-1])
save_dir = os.path.join('./visualize_feature', save_dir)
print('visualize_save_dir: {}'.format(save_dir))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plt.figure(1, figsize=(20, 10))
for label_index in range(len(label_list)):
    plt.subplot(121)
    for i in range(len(seq_type_list)):
        seq_type_mask = np.isin(seq_type, seq_type_list[i]) & np.isin(label, label_list[label_index])
        plt.scatter(tsne_feature[seq_type_mask, 0], tsne_feature[seq_type_mask, 1], label=seq_type_list[i])
    plt.legend(loc="upper left", title="seq_type")
    plt.subplot(122)
    for j in range(len(view_list)):
        view_mask = np.isin(view, view_list[j]) & np.isin(label, label_list[label_index])
        plt.scatter(tsne_feature[view_mask, 0], tsne_feature[view_mask, 1], label=view_list[j])
    plt.legend(loc="upper left", title="view")
    plt.suptitle(label_list[label_index],fontsize=30)
    img_name = os.path.join(save_dir, label_list[label_index])
    plt.savefig(img_name)
    plt.clf()
plt.figure(2, figsize=(40, 40))
for label_index in range(len(label_list)):
    label_mask =  np.isin(label, label_list[label_index])
    plt.scatter(tsne_feature[label_mask, 0], tsne_feature[label_mask, 1], label=label_list[label_index])
plt.legend(loc="upper left", title="label")
img_name = os.path.join(save_dir, 'all')
plt.savefig(img_name)
plt.close("all")