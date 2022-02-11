import numpy as np

# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    assert(acc.shape[0] == acc.shape[1])
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / (acc.shape[1] - 1)
    if not each_angle:
        result = np.mean(result)
    return result

def print_CMC(CMC, config):
    rank_list = np.asarray(config['rank']) - 1
    for i in rank_list:
        print('===Rank-%d (Include identical-view cases)===' % (i + 1))
        if config['dataset'] == 'OUMVLP':
            print(np.mean(CMC[0, :, :, i]))
        else:
            print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                np.mean(CMC[0, :, :, i]),
                np.mean(CMC[1, :, :, i]),
                np.mean(CMC[2, :, :, i])))

    for i in rank_list:
        print('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
        if config['dataset'] == 'OUMVLP':
            print(de_diag(CMC[0, :, :, i]))
        else:
            print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                de_diag(CMC[0, :, :, i]),
                de_diag(CMC[1, :, :, i]),
                de_diag(CMC[2, :, :, i])))
    np.set_printoptions(precision=2, floatmode='fixed')
    for i in rank_list:
        if config['dataset'] == 'OUMVLP':
            print(de_diag(CMC[0, :, :, i], True))
        else:
            print('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
            print('NM:', de_diag(CMC[0, :, :, i], True))
            print('BG:', de_diag(CMC[1, :, :, i], True))
            print('CL:', de_diag(CMC[2, :, :, i], True))
    return [np.mean(CMC[0, :, :, 0]), np.mean(CMC[1, :, :, 0]), np.mean(CMC[2, :, :, 0])]

def print_metric(metric, config, metric_name='mAP'):
    print('==={} (Include identical-view cases)==='.format(metric_name))
    if config['dataset'] == 'OUMVLP':
        print(np.mean(metric[0, :, :]))
    else:
        print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
            np.mean(metric[0, :, :]),
            np.mean(metric[1, :, :]),
            np.mean(metric[2, :, :])))
    print('==={} (Exclude identical-view cases)==='.format(metric_name))
    if config['dataset'] == 'OUMVLP':
        print(de_diag(metric[0, :, :]))
    else:
        print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
            de_diag(metric[0, :, :]),
            de_diag(metric[1, :, :]),
            de_diag(metric[2, :, :])))
    np.set_printoptions(precision=2, floatmode='fixed')
    if config['dataset'] == 'OUMVLP':
        print(de_diag(metric[0, :, :], True))
    else:
        print('==={} of each angle (Exclude identical-view cases)==='.format(metric_name))
        print('NM:', de_diag(metric[0, :, :], True))
        print('BG:', de_diag(metric[1, :, :], True))
        print('CL:', de_diag(metric[2, :, :], True))

# Exclude identical-view cases
def de_diag_exclude_negative(acc, each_angle=False):
    assert(acc.shape[0] == acc.shape[1])
    # result = np.sum(acc - np.diag(np.diag(acc)), 1) / (acc.shape[1] - 1)
    acc = acc - np.diag(np.diag(acc))
    result = []
    for i in range(acc.shape[0]):
        tmp = acc[i, :]
        tmp = tmp[tmp >= 0]
        result.append(np.sum(tmp) / (len(tmp) -1))
    result = np.asarray(result)
    if not each_angle:
        result = np.mean(result)
    return result

def print_metric_exclude_negative(metric, config, metric_name='PRECISION'):
    print('==={} (Include identical-view cases)==='.format(metric_name))
    if config['dataset'] == 'OUMVLP':
        tmp = metric[0, :, :]
        print(np.mean(tmp[tmp >= 0]))
    else:
        tmp0 = metric[0, :, :]
        tmp1 = metric[1, :, :]
        tmp2 = metric[2, :, :]
        print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
            np.mean(tmp0[tmp0 >= 0]),
            np.mean(tmp1[tmp1 >= 0]),
            np.mean(tmp2[tmp2 >= 0])))
    print('==={} (Exclude identical-view cases)==='.format(metric_name))
    if config['dataset'] == 'OUMVLP':
        print(de_diag_exclude_negative(metric[0, :, :]))
    else:
        print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
            de_diag_exclude_negative(metric[0, :, :]),
            de_diag_exclude_negative(metric[1, :, :]),
            de_diag_exclude_negative(metric[2, :, :])))
    np.set_printoptions(precision=2, floatmode='fixed')
    if config['dataset'] == 'OUMVLP':
        print(de_diag(metric[0, :, :], True))
    else:
        print('==={} of each angle (Exclude identical-view cases)==='.format(metric_name))
        print('NM:', de_diag_exclude_negative(metric[0, :, :], True))
        print('BG:', de_diag_exclude_negative(metric[1, :, :], True))
        print('CL:', de_diag_exclude_negative(metric[2, :, :], True))