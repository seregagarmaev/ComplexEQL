"""
Multilayer perceptron for system identification.

This uses regression with square error and L1 norm on weights to get a sparse representation.
"""

import time
import os
import sys
import timeit
import pickle
import getopt
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import utils


__docformat__ = 'restructedtext en'


def logistic(x):
    return 1.0 / (1.0 + torch.exp(-x))


class LinearRegression(nn.Module):
    """Regression layer (linear regression)."""

    def __init__(self, rng, n_in, n_out):
        super().__init__()

        W_values = np.asarray(
            rng.uniform(
                low=-np.sqrt(1.0 / (n_in + n_out)),
                high=np.sqrt(1.0 / (n_in + n_out)),
                size=(n_in, n_out),
            ),
            dtype=np.float32,
        )
        b_values = np.zeros((n_out,), dtype=np.float32)

        self.W = nn.Parameter(torch.tensor(W_values, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b_values, dtype=torch.float32))

    def forward(self, inp):
        return inp @ self.W + self.b

    def get_params(self):
        return [
            self.W.detach().cpu().numpy().copy(),
            self.b.detach().cpu().numpy().copy(),
        ]

    def set_params(self, newParams):
        self.W.data.copy_(torch.tensor(newParams[0], dtype=torch.float32))
        self.b.data.copy_(torch.tensor(newParams[1], dtype=torch.float32))

    def get_weights(self):
        return self.W.detach().cpu().numpy()

    def loss(self, output, y):
        return torch.mean((output - y) ** 2)

    def L1_value(self):
        return torch.abs(self.W).sum()

    def L2_sqr_value(self):
        return torch.sum(self.W ** 2)


class HiddenLayer(nn.Module):
    def __init__(self, rng, n_in, n_units, layer_idx, W=None, b=None):
        super().__init__()

        self.layer_idx = layer_idx
        self.n_out = n_units

        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6.0 / (n_in + self.n_out)),
                    high=np.sqrt(6.0 / (n_in + self.n_out)),
                    size=(n_in, self.n_out),
                ),
                dtype=np.float32,
            )
        else:
            W_values = np.asarray(W, dtype=np.float32)

        if b is None:
            b_values = np.zeros((self.n_out,), dtype=np.float32)
        else:
            b_values = np.asarray(b, dtype=np.float32)

        self.W = nn.Parameter(torch.tensor(W_values, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b_values, dtype=torch.float32))

    def forward(self, inp):
        node_inputs = inp @ self.W + self.b
        return torch.tanh(node_inputs)

    def get_params(self):
        return [
            self.W.detach().cpu().numpy().copy(),
            self.b.detach().cpu().numpy().copy(),
        ]

    def set_params(self, newParams):
        self.W.data.copy_(torch.tensor(newParams[0], dtype=torch.float32))
        self.b.data.copy_(torch.tensor(newParams[1], dtype=torch.float32))

    def get_weights(self):
        return self.W.detach().cpu().numpy()

    def L1_value(self):
        return torch.abs(self.W).sum()

    def L2_sqr_value(self):
        return torch.sum(self.W ** 2)


class MLP(nn.Module):
    """Multi-layer perceptron."""

    def __init__(self, rng, n_in, n_units, n_out, n_layer=1, gradient=None):
        super().__init__()

        self.n_layers = n_layer
        self.n_in = n_in
        self.n_out = n_out
        self.gradient = "sgd" if gradient is None else gradient

        self.hidden_layers = nn.ModuleList()

        for l in range(n_layer):
            if l == 0:
                n_input = n_in
            else:
                n_input = self.hidden_layers[l - 1].n_out

            hiddenLayer = HiddenLayer(
                rng=rng,
                n_in=n_input,
                n_units=n_units,
                layer_idx=l,
            )
            self.hidden_layers.append(hiddenLayer)

        self.output_layer = LinearRegression(
            rng=rng,
            n_in=self.hidden_layers[-1].n_out,
            n_out=n_out,
        )

    def forward_hidden(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return x

    def forward(self, x):
        hidden = self.forward_hidden(x)
        return self.output_layer(hidden)

    def get_params(self):
        params = []
        for layer in self.hidden_layers:
            params.extend(layer.get_params())
        params.extend(self.output_layer.get_params())
        return params

    def get_state(self):
        return [l.get_params() for l in self.hidden_layers] + [self.output_layer.get_params()]

    def set_state(self, newState):
        for s, l in zip(newState, list(self.hidden_layers) + [self.output_layer]):
            l.set_params(s)

    def get_active_units(self, thresh=0.1):
        total = 0
        for layer_idx in range(0, self.n_layers):
            layer = self.hidden_layers[layer_idx]
            in_weights = layer.get_weights()
            out_weights = (
                self.hidden_layers[layer_idx + 1].get_weights()
                if layer_idx + 1 < self.n_layers
                else self.output_layer.get_weights()
            )
            in_weight_norm = np.linalg.norm(in_weights, axis=0, ord=1)
            out_weight_norm = np.linalg.norm(out_weights, axis=1, ord=1)
            total += np.sum((out_weight_norm * in_weight_norm) > thresh * thresh)
        return int(total)

    def get_active_units_old(self, thresh=0.05):
        total = 0
        for layer_idx in range(1, self.n_layers + 1):
            layer = self.hidden_layers[layer_idx] if layer_idx < self.n_layers else self.output_layer
            out_weight_norm = np.linalg.norm(layer.get_weights(), axis=1, ord=1)
            total += np.sum(out_weight_norm > thresh)
        return int(total)

    def evaluate(self, input):
        x = torch.tensor(utils.cast_to_floatX(input), dtype=torch.float32)
        with torch.no_grad():
            y = self.forward(x)
        return y.detach().cpu().numpy()

    def L1_value(self):
        return self.output_layer.L1_value() + sum([l.L1_value() for l in self.hidden_layers])

    def L2_sqr_value(self):
        return self.output_layer.L2_sqr_value() + sum([l.L2_sqr_value() for l in self.hidden_layers])

    def _make_optimizer(self, learning_rate):
        if self.gradient == 'sgd':
            return optim.SGD(self.parameters(), lr=learning_rate)
        raise ValueError("unknown gradient " + str(self.gradient))

    def train_step(self, input, labels, L1_reg, L2_reg, learning_rate):
        self.train()

        x = torch.tensor(utils.cast_to_floatX(input), dtype=torch.float32)
        y = torch.tensor(utils.cast_to_floatX(labels), dtype=torch.float32)

        optimizer = self._make_optimizer(learning_rate)
        optimizer.zero_grad()

        output = self.forward(x)
        cost = self.output_layer.loss(output, y) + L1_reg * self.L1_value() + L2_reg * self.L2_sqr_value()

        cost.backward()
        optimizer.step()

        return float(cost.detach().cpu().item())

    def test_model(self, input, labels):
        self.eval()

        x = torch.tensor(utils.cast_to_floatX(input), dtype=torch.float32)
        y = torch.tensor(utils.cast_to_floatX(labels), dtype=torch.float32)

        with torch.no_grad():
            output = self.forward(x)
            return float(self.output_layer.loss(output, y).detach().cpu().item())

    def validate_model(self, input, labels):
        return self.test_model(input, labels)


def test_mlp(
    datasets,
    learning_rate=0.01,
    L1_reg=0.001,
    L2_reg=0.00,
    n_epochs=200,
    batch_size=20,
    n_layer=1,
    n_units=30,
    classifier=None,
    init_state=None,
    gradient=None,
    verbose=True,
    param_store=None,
    id=None,
    validate_every=50,
    reg_start=0,
    reg_end=None,
):
    train_set_x, train_set_y = utils.cast_dataset_to_floatX(datasets[0])
    valid_set_x, valid_set_y = utils.cast_dataset_to_floatX(datasets[1])

    if len(datasets) > 2 and len(datasets[2]) == 2:
        test_set_x, test_set_y = utils.cast_dataset_to_floatX(datasets[2])
        n_test_batches = test_set_x.shape[0] // batch_size
    else:
        test_set_x = test_set_y = None
        n_test_batches = 0

    n_train_batches = train_set_x.shape[0] // batch_size
    n_valid_batches = valid_set_x.shape[0] // batch_size

    inputdim = len(datasets[0][0][0])
    outputdim = len(datasets[0][1][0])

    if verbose:
        print("Input/output dim:", (inputdim, outputdim))
    if verbose and test_set_x is not None:
        print("Training set, test set:", (train_set_x.shape[0], test_set_x.shape[0]))
    elif verbose:
        print("Training set:", train_set_x.shape[0])

    print('... building the model')

    rng = np.random.RandomState(int(time.time()) if id is None else id)
    if classifier is None:
        classifier = MLP(
            rng=rng,
            n_in=inputdim,
            n_units=n_units,
            n_out=outputdim,
            n_layer=n_layer,
            gradient=gradient,
        )

    if init_state:
        classifier.set_state(init_state)

    print('... training')
    sys.stdout.flush()

    improvement_threshold = 0.99

    best_validation_error = np.inf
    this_validation_error = np.inf
    best_epoch = 0
    test_score = 0.0
    best_state = classifier.get_state()

    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    train_errors = []
    validation_errors = []
    test_errors = []

    if param_store is not None:
        param_store.append(classifier.get_params())

    if reg_end is None:
        reg_end = 0

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        reg_factor = 0.0

        if reg_start < epoch <= reg_end:
            reg_factor = 1.0

        minibatch_avg_cost = 0.0

        for minibatch_index in range(n_train_batches):
            index = minibatch_index
            minibatch_avg_cost = classifier.train_step(
                input=train_set_x[index * batch_size: (index + 1) * batch_size],
                labels=train_set_y[index * batch_size: (index + 1) * batch_size],
                L1_reg=L1_reg * reg_factor,
                L2_reg=L2_reg * reg_factor,
                learning_rate=learning_rate,
            )

        train_errors.append([epoch, minibatch_avg_cost])

        if param_store is not None:
            param_store.append(classifier.get_params())

        if epoch == 1 or epoch % validate_every == 0 or epoch == n_epochs:
            this_validation_errors = [
                classifier.validate_model(
                    input=valid_set_x[index * batch_size:(index + 1) * batch_size],
                    labels=valid_set_y[index * batch_size:(index + 1) * batch_size],
                )
                for index in range(n_valid_batches)
            ]
            this_validation_error = float(np.mean(this_validation_errors))
            validation_errors.append([epoch, this_validation_error])

            if verbose:
                print(
                    'epoch %i, minibatch %i/%i, minibatch_avg_cost %f validation error %f'
                    % (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        minibatch_avg_cost,
                        this_validation_error,
                    )
                )

            if test_set_x is not None:
                test_losses = [
                    classifier.test_model(
                        input=test_set_x[index * batch_size:(index + 1) * batch_size],
                        labels=test_set_y[index * batch_size:(index + 1) * batch_size],
                    )
                    for index in range(n_test_batches)
                ]
                this_test_score = float(np.mean(test_losses))
                test_errors.append([epoch, this_test_score])
            else:
                this_test_score = np.inf

            if this_validation_error < best_validation_error:
                if this_validation_error < best_validation_error * improvement_threshold:
                    best_state = classifier.get_state()

                best_validation_error = this_validation_error
                best_epoch = epoch
                test_score = this_test_score

                if verbose:
                    print(
                        '     epoch %i, minibatch %i/%i, test error of best model %f'
                        % (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score,
                        )
                    )

        if epoch % 1000 == 0:
            print("Epoch: ", epoch, " Best val error: ", best_validation_error)
            sys.stdout.flush()

    end_time = timeit.default_timer()
    time_required = (end_time - start_time) / 60.0

    print(
        'Optimization complete. Best validation score of %f obtained at epoch %i, with test performance %f '
        % (best_validation_error, best_epoch + 1, test_score)
    )
    print(
        'The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % time_required,
        file=sys.stderr,
    )

    if verbose:
        np.set_printoptions(precision=4, suppress=True)
        print(classifier.get_params())

    return {
        'train_losses': np.asarray(train_errors),
        'val_errors': np.asarray(validation_errors),
        'test_errors': np.asarray(test_errors),
        'classifier': classifier,
        'test_score': test_score,
        'val_score': this_validation_error,
        'best_val_score': best_validation_error,
        'best_epoch': best_epoch,
        'best_state': best_state,
        'num_active': classifier.get_active_units(),
        'runtime': time_required,
    }


def usage():
    print(
        sys.argv[0] + "[-i id -d dataset -p extrapolationdataset -l layers -e epochs -n nodes"
        + " -r learningrate --l1=l1reg --l2=l2reg --shortcut --resfolder"
        + "  --gradient=sgd|adam --initfile=statefile -v -o]"
    )


if __name__ == "__main__":
    dataset_file = None
    extra_pol_test_sets = []
    extra_pols = []
    n_epochs = 1200
    n_layers = 3
    n_nodes = 10
    batch_size = 20
    init_file = None
    init_state = None
    gradient = "sgd"
    L1_reg = 0.00001
    L2_reg = 0.00001
    learning_rate = 0.01
    reg_start = 0
    reg_end = None
    output = False
    verbose = 0
    id = np.random.randint(0, 1000000)
    result_folder = "./"

    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hv:i:d:p:l:e:n:f:co",
            [
                "help", "verbose=", "id=", "dataset=", "extrapol=", "layers=", "epochs=",
                "nodes=", "l1=", "l2=", "lr=", "resfolder=", "batchsize=", "initfile=",
                "gradient=", "reg_start=", "reg_end=", "output"
            ]
        )
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-v", "--verbose"):
            verbose = int(arg)
        elif opt in ("-i", "--id"):
            id = int(arg)
        elif opt in ("-d", "--dataset"):
            dataset_file = arg
        elif opt in ("-p", "--extrapol"):
            extra_pol_test_sets.append(arg)
        elif opt in ("-l", "--layers"):
            n_layers = int(arg)
        elif opt in ("-e", "--epochs"):
            n_epochs = int(arg)
        elif opt in ("--batchsize"):
            batch_size = int(arg)
        elif opt in ("--l1"):
            L1_reg = float(arg)
        elif opt in ("--l2"):
            L2_reg = float(arg)
        elif opt in ("--lr"):
            learning_rate = float(arg)
        elif opt in ("-n", "--nodes"):
            n_nodes = int(arg)
        elif opt in ("--initfile"):
            init_file = arg
        elif opt in ("--gradient"):
            gradient = arg
        elif opt in ("--reg_start"):
            reg_start = int(arg)
        elif opt in ("--reg_end"):
            reg_end = int(arg)
        elif opt in ("-o", "--output"):
            output = True
        elif opt in ("-f", "--resfolder"):
            result_folder = arg

    if not dataset_file:
        print("provide datasetfile!")
        usage()
        exit(1)

    dataset = utils.load_data(dataset_file)

    if len(extra_pol_test_sets) > 0:
        if verbose > 0:
            print("do also extrapolation test(s)!")
        extra_pols = [utils.load_data(test_set) for test_set in extra_pol_test_sets]

    if init_file:
        with open(init_file, 'rb') as f:
            init_state = pickle.load(f, encoding='latin1')
            print("load initial state from file " + init_file)

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    name = result_folder + str(id)

    result = test_mlp(
        datasets=dataset,
        n_epochs=n_epochs,
        verbose=verbose > 0,
        learning_rate=learning_rate,
        L1_reg=L1_reg,
        L2_reg=L2_reg,
        n_layer=n_layers,
        n_units=n_nodes,
        id=id,
        gradient=gradient,
        batch_size=batch_size,
        init_state=init_state,
        reg_start=reg_start,
        reg_end=reg_end,
    )

    classifier = result['classifier']

    with open(name + '.best_state', 'wb') as f:
        pickle.dump(result['best_state'], f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(name + '.last_state', 'wb') as f:
        pickle.dump(classifier.get_state(), f, protocol=pickle.HIGHEST_PROTOCOL)

    extra_scores = []
    extra_scores_best = []

    for extra in extra_pols:
        extra_set_x, extra_set_y = extra[0]
        extra_scores.append(classifier.test_model(input=extra_set_x, labels=extra_set_y))

    classifier.set_state(result['best_state'])

    for extra in extra_pols:
        extra_set_x, extra_set_y = extra[0]
        extra_scores_best.append(classifier.test_model(input=extra_set_x, labels=extra_set_y))

    result_line = ""
    with open(name + '.res', 'w') as f:
        if id == 0:
            f.write(
                '#C layers epochs nodes lr L1 L2 batchsize regstart regend'
                + ' id dataset gradient numactive bestepoch runtime'
                + "".join([' extrapol' + str(i) for i in range(1, len(extra_scores) + 1)])
                + "".join([' extrapolbest' + str(i) for i in range(1, len(extra_scores_best) + 1)])
                + ' valerror valerrorbest testerror\n'
            )
            f.write('# extra datasets: ' + " ".join(extra_pol_test_sets) + '\n')

        result_line = [
            str(n_layers),
            str(n_epochs),
            str(n_nodes),
            str(learning_rate),
            str(L1_reg),
            str(L2_reg),
            str(batch_size),
            str(reg_start),
            str(reg_end),
            str(id),
            dataset_file,
            gradient,
            str(result['num_active']),
            str(result['best_epoch']),
            str(result['runtime']),
        ] + [str(e) for e in extra_scores] + [str(e) for e in extra_scores_best] + [
            str(result['val_score']),
            str(result['best_val_score']),
            str(result['test_score']),
        ]

        f.write(str.join('\t', result_line) + '\n')

    with open(name + '.validerror', 'w', newline='') as csvfile:
        a = csv.writer(csvfile, delimiter='\t')
        a.writerows([["#C epoch", "val_error"]])
        a.writerows([["# "] + result_line])
        a.writerows(result['val_errors'])

    if output:
        with open(name + '.trainloss', 'w', newline='') as csvfile:
            a = csv.writer(csvfile, delimiter='\t')
            a.writerows([["#C epoch", "train_loss"]])
            a.writerows([["# "] + result_line])
            a.writerows(result['train_losses'])

        if len(result['test_errors']) > 0:
            with open(name + '.testerrors', 'w', newline='') as csvfile:
                a = csv.writer(csvfile, delimiter='\t')
                a.writerows([["#C epoch", "test_error"]])
                a.writerows([["# "] + result_line])
                a.writerows(result['test_errors'])