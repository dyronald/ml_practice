import time
import numpy as np
from neural_net.network import Network
import neural_net.layer_loop as ll
import neural_net.layer_matrix as lm

"""
Compare the 2 layers implementations.
Output of each method are expected to be identical.
"""

if __name__ == '__main__':    
    X = np.array([
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 0],
        [1, 0, 0],
    ])
    y = np.array([[0, 1, 1, 0, 0, 0]]).T
    num_features = X.shape[1]

    l_layer1 = ll.Layer(
        nodes = [
            ll.LogisticNode(num_features=num_features,),
            ll.LogisticNode(num_features=num_features,),
            ll.LogisticNode(num_features=num_features,),
            ll.LogisticNode(num_features=num_features,),
            ll.LogisticNode(num_features=num_features,),
       ])
    l_layer2 = ll.Layer(
        nodes = [
            ll.LogisticNode(num_features=5,),
            ll.LogisticNode(num_features=5,),
            ll.LogisticNode(num_features=5,),
            ll.LogisticNode(num_features=5,),
        ])
    l_layer3 = ll.OutputLayer(
        nodes = [
            ll.LogisticNode(num_features=4,),
        ])

    l1 = lm.Layer(num_features, 5)
    l2 = lm.Layer(5, 4)
    l3 = lm.OutputLayer(4, 1)

    l1.thetas = l_layer1.thetas()
    l2.thetas = l_layer2.thetas()
    l3.thetas = l_layer3.thetas()

    old_net = Network(
        layers = [l_layer1, l_layer2, l_layer3],
        features = X,
        labels = y,
    )
    old_start = time.perf_counter()
    print('training loop layers')
    old_net.train(15)
    old_time = time.perf_counter() - old_start

    new_net = Network(
        layers = [l1, l2, l3],
        features = X,
        labels = y,
    )
    new_start = time.perf_counter()
    print('training matrix layers')
    new_net.train(15)
    new_time = time.perf_counter() - new_start

    old_hyp = old_net.hypothesis()
    new_hyp = new_net.hypothesis()
    print('')
    print(f'old: {old_hyp}')
    print(f'new: {new_hyp}')
    print(f'equal: {old_hyp == new_hyp}')
    print(f'old time: {old_time}')
    print(f'new time: {new_time}')

    print('--- a1 ---')
    print(X[2])
    l_a1 = l_layer1.activation(X[2])
    a1 = l1.activation(X[2])
    print(l_a1)
    print(a1)
    print(l_a1 == a1)

    print('--- a2 ---')
    l_a2 = l_layer2.activation(a1)
    a2 = l2.activation(a1)
    print(l_a2)
    print(a2)
    print(l_a2 == a2)

    print('--- a3 ---')
    l_a3 = l_layer3.activation(a2)
    a3 = l3.activation(a2)
    print(l_a3)
    print(a3)
    print(l_a3 == a3)

    print('--- error3 ---')
    l_error3 = l_layer3.calc_error(1)
    error3 = l3.calc_error(1)
    print(l_error3)
    print(error3)
    print(l_error3 == error3)

    print('--- error2 ---')
    l_error2 = l_layer2.calc_error(l_layer3)
    error2 = l2.calc_error(l3)
    print(l_error2)
    print(error2)
    print(l_error2 == error2)

    print('--- error1 ---')
    l_error1 = l_layer1.calc_error(l_layer2)
    error1 = l1.calc_error(l2)
    print(l_error1)
    print(error1)
    print(l_error1 == error1)

    print('--- delta3 ---')
    print(l_layer3.accumulate_delta())
    print(l3.accumulate_delta())

    print('--- delta2 ---')
    print(l_layer2.accumulate_delta())
    print(l2.accumulate_delta())

    print('--- delta1 ---')
    print(l_layer1.accumulate_delta())
    print(l1.accumulate_delta())

    print('-- step --')
    print('--- nn ---')
    l_layer3.step_theta()
    l_layer2.step_theta()
    l_layer1.step_theta()
    print('--- matrix ---')
    l3.step_theta()
    l2.step_theta()
    l1.step_theta()

    print('--- theta3 ---')
    print(l_layer3.thetas())
    print(l3.thetas)
    print(l_layer3.thetas() == l3.thetas)

    print('--- theta2 ---')
    print(l_layer2.thetas())
    print(l2.thetas)
    print(l_layer2.thetas() == l2.thetas)

    print('--- theta1 ---')
    print(l_layer1.thetas())
    print(l1.thetas)
    print(l_layer1.thetas() == l1.thetas)
