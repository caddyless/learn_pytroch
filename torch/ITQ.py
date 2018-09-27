import numpy as np
from termcolor import colored


def relu(x):
    return np.maximum(x, 0.)

def space(*kwargs):
    strarr = arr2strarr(*kwargs)
    return ' '.join(strarr)


def arr2strarr(*kwargs):
    mylist = []
    for i in kwargs:
        if isinstance(i, str):
            mylist.append(i)
            continue
        mylist.append(str(i))
    return mylist


def epscheck(x, tol=5):
    tmp = np.any(np.abs(x) > 10**tol)
    if tmp:
        redprint('1e'+str(tol)+' exceed')

def red(sentence):
    return colored(sentence, 'red')

def redprint(sentence):
    print(red(space(sentence)))


def ITQ_decompose(
        feature,
        gt_feature,
        weight,
        rank,
        bias=None,
        DEBUG=False,
        Wr=None):

    n_ins = feature.shape[0]
    n_filter_channels = feature.shape[1]
    assert gt_feature.shape[0] == n_ins
    assert gt_feature.shape[1] == n_filter_channels

    # do itq
    if 0:
        Y_div = n_ins * feature.std()  # * n_filter_channels
        Y = feature.copy() / Y_div
        # r(yi)
        Z = relu(gt_feature) / Y_div
    else:
        Y = feature.copy()
        # r(yi)
        Z = relu(gt_feature)
    Zsq = Z**2
    Y_mean = Y.mean(0)
    # Y
    G = Y - Y_mean
    # (Y'Y)^-1
    PG = (G.T).dot(G)
    if 0:
        print(np.linalg.cond(PG))
        embed()
    PG = pinv(PG)
    # epscheck(PG)
    # epscheck(PG,6)
    # epscheck(PG,7)
    # epscheck(PG,8)
    # epscheck(PG,9)
    # epscheck(PG,10)

    PGGt = PG.dot(G.T)

    # init U as Y
    UU = G.copy()
    U_mean = Y_mean.copy()
    if 1:
        print("Reconstruction Err", rel_error(Z, relu(Y)))

    lambdas = [0.1, 1]
    step_iters = [30, 20]  # , 20, 20, 20
    for step in range(len(lambdas)):
        Lambda = lambdas[step]
        for iter in range(step_iters[step]):

            #  TODO    Y * (Y'Y)^-1 * (Y'*Z)
            #X = G.dot(Ax_b(PG, (G.T).dot(UU)))
            #X = G.dot(Ax_b(G, UU))
            if 0:
                epscheck(X, 10)
            X = G.dot(PGGt.dot(UU))
            #X = G.dot(PG.dot((G.T).dot(UU)))

            L, sigma, R = svd(X)
            #L, sigma, R = np.linalg.svd(X, 0)

            T = L[:, :rank].dot(np.diag(sigma[:rank])).dot(R[:rank, :])
            if 0:
                print("RX", error(X, T))

            #T = Ax_b(PG, (G.T).dot(T))
            #T = Ax_b(G, T)
            if 0:
                epscheck(T, 10)
            T = PGGt.dot(T)
            #T = PG.dot((G.T).dot(T))
            RU = G.dot(T)
            if 0:
                print("RU", rel_error(UU, RU))

            RU += U_mean
            # case 0: U <= 0
            U0 = np.minimum(RU, 0.)
            Cost0 = Zsq + Lambda * (U0 - RU)**2

            # case 1: U > 0
            U1 = relu((Lambda * RU + Z) / (Lambda + 1.))
            Cost1 = (U1 - Z)**2 + Lambda * (U1 - RU)**2

            U = (Cost0 <= Cost1) * U0 + (Cost0 > Cost1) * U1

            U_mean = U.mean(0)
            # Z
            UU = U - U_mean
            if DEBUG:
                loss = error(Z, relu(U))
                print("loss", loss, "rel", rel_error(Z, relu(U)))

    # process output
    L, sigma, R = svd(T)
    #L, sigma, R = np.linalg.svd(T,0)
    L = L[:, :rank]
    R = np.diag(sigma[:rank]).dot(R[:rank, :])

    dim = weight.shape
    W12 = np.ndarray(
        Wr.shape if Wr is not None else weight.shape,
        weight.dtype)
    assert len(dim) == 4

    right = 1

    if dim[3] != n_filter_channels:
        assert dim[0] == n_filter_channels
        if right:
            weight = np.transpose(weight, [1, 2, 3, 0])
            W1 = weight.reshape([-1, n_filter_channels]).dot(L)
        else:
            W1 = R.dot(weight.reshape([n_filter_channels, -1]))

        if Wr is not None:
            Wr = np.transpose(Wr, [1, 2, 3, 0])
            W12 = Wr.reshape([-1, n_filter_channels]).dot(L)
        else:
            W12 = W1

        if 0:
            # embed()
            print(W1.shape)
            print(weight.shape)
            print(dim)
        if right:
            W1 = W1.reshape(weight.shape[:3] + (rank,))
            W1 = np.transpose(W1, [3, 0, 1, 2])
        else:
            W1 = W1.reshape((rank,) + W1.shape[1:])
    else:
        assert False
        W1 = weight.reshape([-1, n_filter_channels]).dot(L)
        W1 = W1.reshape([dim[0], dim[1], dim[2], rank])

    W2 = R
    if right:
        W12 = W12.dot(W2)
    else:
        W12 = W2.T.dot(W12)
    W2 = W2.T
    W2 = W2.reshape([n_filter_channels, rank, 1, 1])

    if right:
        W12 = W12.reshape(
            (Wr.shape[:3] if Wr is not None else weight.shape[:3]) + (n_filter_channels,))
        W12 = np.transpose(W12, [3, 0, 1, 2])
    else:
        W12 = W12.reshape(dim)

    B = - Y_mean.dot(T) + U_mean
    if bias is not None:
        B = B.T + bias
    else:
        B = B.T

    if 1:
        epscheck(W1, 2)
        epscheck(W2, 2)
        epscheck(B, 2)
        epscheck(W12, 2)
        epscheck(W1, 4)
        epscheck(W2, 4)
        epscheck(B, 4)
        epscheck(W12, 4)
    return W1, W2, B, W12
