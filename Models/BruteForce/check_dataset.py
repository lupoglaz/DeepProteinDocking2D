from DeepProteinDocking2D.Models.BruteForce.utility_functions import *

if __name__ == '__main__':
    data = read_pkl('toy_concave_data/interaction_data_train')

    print(data[0][0])
    print(data[0][1])
    print(data[0][2])

    # for i in range(len(data)):
    #     print(data[i])
    # print(len(data[0]))
    # print(data[0])
    # for i in range(len(data[0])):
    #     print(data[0][i].shape)
        # plt.imshow(data[i][0])
        # plt.show()