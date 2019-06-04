import numpy as np
import copy


def create_list_waights(size_hidden_layer1):
    list_waights = []
    waight_h = [(2 * np.random.random((size_hidden_layer1, size_hidden_layer1)) - 1)]
    list_waights.append(2*np.random.random((2, size_hidden_layer1)) - 1)
    list_waights.append(2 * np.random.random((size_hidden_layer1, 1)) - 1)
    return list_waights, waight_h


def main(size_hidden_layer, Alpha):
    waights_list, h_waight = create_list_waights(size_hidden_layer)
    waights1_update = np.zeros_like(waights_list[1])
    waights0_update = np.zeros_like(waights_list[0])
    h_waight_update = np.zeros_like(h_waight)
    for n in range(100000):
        rand1 = np.random.randint(127)
        rand2 = np.random.randint(128)
        res = rand1+rand2
        data_a = [int(x) for x in '{0:08b}'.format(rand1)]
        data_b = [int(x) for x in '{0:08b}'.format(rand2)]
        y1 = [int(x) for x in '{0:08b}'.format(res)]
        layer_1_values = [np.zeros(size_hidden_layer)]
        layer2_delta = []
        overallError = 0
        d = np.zeros_like(y1)
        for m in range(len(data_a)):
            X = np.array([[data_a[-m-1], data_b[-m-1]]])
            Y = np.array([y1[-m-1]]).reshape(1, 1)
            layer1 = (1/(1+np.exp(-(np.dot(X, waights_list[0]) + np.dot(layer_1_values[-1], h_waight))))).reshape(1, size_hidden_layer)
            layer2 = (1/(1+np.exp(-(np.dot(layer1, waights_list[1]))))).reshape(1, 1)
            layer2_delta.append((Y - layer2)*(layer2*(1-layer2)))
            layer_1_values.append(copy.deepcopy(layer1))
            overallError += ((Y - layer2)[0][0])
            d[-m-1] = np.round(layer2)
        future_layer_1_delta = np.zeros(size_hidden_layer)
        for q in range(len(data_a)):
            X = np.array([[data_a[q], data_b[q]]])
            layer_1_delta = ((future_layer_1_delta.dot(h_waight) + layer2_delta[-q-1].dot(
                waights_list[1].T)) * (layer_1_values[-q-1]*(1-layer_1_values[-q-1]))).reshape(1, size_hidden_layer)
            future_layer_1_delta = layer_1_delta
            waights1_update += np.atleast_2d(layer_1_values[-q-1]).T.dot(layer2_delta[-q-1])
            waights0_update += X.T.dot(layer_1_delta)
            h_waight_update += np.atleast_2d(layer_1_values[-q-2]).T.dot(layer_1_delta)
        waights_list[1] += waights1_update * Alpha
        waights_list[0] += waights0_update * Alpha
        h_waight += h_waight_update * Alpha
        waights1_update *= 0
        waights0_update *= 0
        h_waight_update *= 0
        if n % 1000 == 0:
            print("Error:" + str(overallError))
            print("Pred:" + str(d))
            print("True:" + str(y1))
    return


main(16, 0.07)



