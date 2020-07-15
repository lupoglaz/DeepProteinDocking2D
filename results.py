import matplotlib.pylab as plt
import seaborn as sea
sea.set_style("whitegrid")

def load_log(filename):
    with open(filename) as fin:
        header = fin.readline()
        loss = [tuple([float(x) for x in line.split()]) for line in fin]
    return list(zip(*loss))


if __name__=='__main__':
    # f = plt.figure()
    # train = load_log('Log/log_train_spatial_rotzero.txt')
    # valid = load_log('Log/log_valid_spatial_rotzero.txt')
    # plt.plot(train[1], label='train')
    # plt.plot(valid[1], label='valid')
    # plt.legend()
    # # plt.show()
    # plt.savefig('training_spatial.png')

    # f = plt.figure()
    # train = load_log('Log/log_train_spatial_eq.txt')
    # valid = load_log('Log/log_valid_spatial_eq.txt')
    # plt.plot(train[1], label='train')
    # plt.plot(valid[1], label='valid')
    # plt.ylim([0,1.0])
    # plt.legend()
    # # plt.show()
    # plt.savefig('training_spatial_eq.png')

    # f = plt.figure()
    # train = load_log('Log/log_train_global.txt')
    # valid = load_log('Log/log_valid_global.txt')
    # plt.plot(train[1], label='train')
    # plt.plot(valid[1], label='valid')
    # plt.legend()
    # # plt.show()
    # plt.savefig('training_global_correct.png')

    f = plt.figure()
    train = load_log('Log/log_train_global_eq.txt')
    valid = load_log('Log/log_valid_global_eq.txt')
    plt.plot(train[1], label='train')
    plt.plot(valid[1], label='valid')
    plt.ylim([0,1.2])
    plt.legend()
    plt.show()
    # plt.savefig('training_global_equivariant.png')