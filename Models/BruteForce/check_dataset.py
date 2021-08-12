from DeepProteinDocking2D.Models.BruteForce.utility_functions import *

if __name__ == '__main__':
    data = read_pkl('toy_concave_data/interaction_data_train')

    print(len(data))
    print(len(data[0]))
    print(len(data[1]))

    print(data[0][0])
    print(data[1][0])
    print(data[1][1])

    # print(len(data[0][0]))
    # print(len(data[1][0]))


    # for i in range(len(data[0])):
    #     plt.imshow(data[0][i])
    #     plt.show()

    # for i in range(len(data[1])):
    #     plt.imshow(data[1][i])
    #     plt.show()