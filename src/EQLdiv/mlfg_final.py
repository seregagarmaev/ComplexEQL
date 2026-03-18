import os
import csv
import time
import sys
import timeit
import getopt
import random
import pickle

import numpy as np
import sympy as sp
import torch
import torch.nn as nn
import torch.optim as optim

from src.EQLdiv.utils import *


__docformat__ = 'restructedtext en'


class LinearRegression(nn.Module):
    """
    Regression layer with division in the output layer.
    """

    def __init__(self, rng, n_in, n_out, div_thresh=1e-4):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.div_thresh = div_thresh

        W_values = rng.normal(
            loc=0.0,
            scale=np.sqrt(1.0 / (n_in + 2 * n_out)),
            size=(n_in, 2 * n_out),
        ).astype(np.float32)

        b_values = np.ones((2 * n_out,), dtype=np.float32)

        self.W = nn.Parameter(torch.tensor(W_values, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b_values, dtype=torch.float32))

    def activation(self, x, thresh):
        return torch.where(
            x < thresh,
            torch.zeros_like(x),
            1.0 / torch.clamp(x, min=thresh),
        )

    def pre_output(self, inp):
        return inp @ self.W + self.b

    def forward(self, inp, div_thresh=None):
        if div_thresh is None:
            div_thresh = self.div_thresh

        node_inputs = self.pre_output(inp)
        numerator = node_inputs[:, 0:self.n_out]
        denominator = node_inputs[:, self.n_out:2 * self.n_out]
        output = self.activation(denominator, div_thresh) * numerator

        return output, denominator

    def get_params(self):
        return [
            self.W.detach().cpu().numpy().copy(),
            self.b.detach().cpu().numpy().copy(),
        ]

    def set_params(self, newParams):
        self.W.data.copy_(torch.tensor(newParams[0], dtype=torch.float32))
        self.b.data.copy_(torch.tensor(newParams[1], dtype=torch.float32))

    def get_state(self):
        return self.get_params()

    def set_state(self, newState):
        self.set_params(newState)

    def get_weights(self):
        return self.W.detach().cpu().numpy()

    def set_out_weights(self, row, vec):
        self.W.data[row, :] = torch.tensor(vec, dtype=torch.float32)

    def loss(self, output, y):
        return torch.mean((output - y) ** 2)

    def L1_value(self):
        return torch.abs(self.W).sum() + 0.01 * torch.abs(self.b).sum()

    def L2_sqr_value(self):
        return torch.sum(self.W ** 2) + 0.01 * torch.sum(self.b ** 2)

    def penalty_value(self, denominator, div_thresh):
        return torch.sum((div_thresh - denominator) * (denominator < div_thresh).float())

    def extrapol_loss_value(self, output, denominator, div_thresh):
        return torch.sum(
            (torch.abs(output) - 100.0) * (torch.abs(output) > 100.0).float()
            + (div_thresh - denominator) * (denominator < div_thresh).float()
        )


class FGLayer(nn.Module):
    def __init__(
        self,
        rng,
        n_in,
        n_per_base,
        layer_idx,
        basefuncs1=None,
        basefuncs2=None,
    ):
        super().__init__()

        if basefuncs1 is None:
            basefuncs1 = [0, 1, 2]
        if basefuncs2 is None:
            basefuncs2 = [0]

        self.basefuncs1 = basefuncs1
        self.basefuncs2 = basefuncs2
        self.basefuncs1_uniq = list(set(basefuncs1))
        self.n_basefuncs1_uniq = len(self.basefuncs1_uniq)
        self.n_per_base = n_per_base
        self.funcs1 = ['id', 'sin', 'cos']
        self.funcs2 = ['mult']
        self.layer_idx = layer_idx

        self.n_base1 = len(basefuncs1)
        self.n_base2 = len(basefuncs2)
        n_out = (self.n_base1 + self.n_base2) * n_per_base
        n_w_out = (self.n_base1 + 2 * self.n_base2) * n_per_base
        self.n_out = n_out

        W_values = rng.normal(
            loc=0.0,
            scale=np.sqrt(1.0 / (n_in + n_w_out)),
            size=(n_in, n_w_out),
        ).astype(np.float32)
        b_values = np.zeros((n_w_out,), dtype=np.float32)

        self.W = nn.Parameter(torch.tensor(W_values, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b_values, dtype=torch.float32))

        self.nodes_type1 = np.asarray(np.repeat(basefuncs1, n_per_base), dtype=np.int32)
        self.nodes_type2 = np.asarray(np.repeat(basefuncs2, n_per_base), dtype=np.int32)

    def forward(self, inp):
        node_inputs = inp @ self.W + self.b

        z = node_inputs[:, :self.n_per_base * self.n_base1]
        z1 = node_inputs[
            :,
            self.n_per_base * self.n_base1:self.n_per_base * (self.n_base1 + self.n_base2),
        ]
        z2 = node_inputs[:, self.n_per_base * (self.n_base1 + self.n_base2):]

        fun1 = []
        for i, typ in enumerate(self.nodes_type1):
            zi = z[:, i]
            if typ == 0:
                fun1.append(zi)
            elif typ == 1:
                fun1.append(torch.sin(zi))
            else:
                fun1.append(torch.cos(zi))

        fun2 = []
        for i, typ in enumerate(self.nodes_type2):
            if typ == 0:
                fun2.append(z1[:, i] * z2[:, i])
            else:
                fun2.append(z1[:, i])

        return torch.stack(fun1 + fun2, dim=1)

    def get_params(self):
        return [
            self.W.detach().cpu().numpy().copy(),
            self.b.detach().cpu().numpy().copy(),
        ]

    def set_params(self, newParams):
        self.W.data.copy_(torch.tensor(newParams[0], dtype=torch.float32))
        self.b.data.copy_(torch.tensor(newParams[1], dtype=torch.float32))

    def get_state(self):
        return self.get_params() + [self.nodes_type1.copy(), self.nodes_type2.copy()]

    def set_state(self, newState):
        self.set_params(newState)
        if len(newState) > 2:
            self.nodes_type1 = np.asarray(newState[2], dtype=np.int32)
            self.nodes_type2 = np.asarray(newState[3], dtype=np.int32)
        else:
            print("Not full reload: missing node-types")

    def get_n_type1(self):
        return self.n_base1 * self.n_per_base

    def get_n_type2(self):
        return self.n_base2 * self.n_per_base

    def get_weights(self):
        return self.W.detach().cpu().numpy()

    def get_in_weights(self, idx):
        return self.W[:, idx].detach().cpu().numpy()

    def set_out_weights(self, row, vec):
        self.W.data[row, :] = torch.tensor(vec, dtype=torch.float32)

    def set_in_weights(self, col, vec):
        self.W.data[:, col] = torch.tensor(vec, dtype=torch.float32)

    def get_bias(self, idx):
        return float(self.b[idx].detach().cpu().item())

    def set_bias(self, idx, value):
        self.b.data[idx] = torch.tensor(value, dtype=torch.float32)

    def get_nodes_type1(self):
        return self.nodes_type1

    def get_nodes_type2(self):
        return self.nodes_type2

    def get_node_type1(self, idx):
        return int(self.nodes_type1[idx])

    def set_node_type1(self, idx, typ):
        self.nodes_type1[idx] = typ

    def getNodeFunctions(self, withnumbers=True):
        def name(func, idx):
            if withnumbers:
                return func + '-' + str(self.layer_idx) + '-' + str(idx)
            else:
                return func

        return [
            name(self.funcs1[bf], i)
            for (i, bf) in zip(range(1, len(self.get_nodes_type1()) + 1), self.get_nodes_type1())
        ] + [
            name(self.funcs2[bf], i)
            for (i, bf) in zip(range(1, len(self.get_nodes_type2()) + 1), self.get_nodes_type2())
        ]

    def getWeightCorrespondence(self):
        def name(func, idx):
            return func + '-' + str(self.layer_idx) + '-' + str(idx)

        return [
            name(self.funcs1[bf], i)
            for (i, bf) in zip(range(1, len(self.get_nodes_type1()) + 1), self.get_nodes_type1())
        ] + [
            name(self.funcs2[bf], i) + ':' + '1'
            for (i, bf) in zip(range(1, len(self.get_nodes_type2()) + 1), self.get_nodes_type2())
        ] + [
            name(self.funcs2[bf], i) + ':' + '2'
            for (i, bf) in zip(range(1, len(self.get_nodes_type2()) + 1), self.get_nodes_type2())
        ]

    def L1_value(self):
        return torch.abs(self.W).sum() + 0.01 * torch.abs(self.b).sum()

    def L2_sqr_value(self):
        return torch.sum(self.W ** 2) + 0.01 * torch.sum(self.b ** 2)


class MLFG(nn.Module):
    """
    Multi-Layer Function Graph
    """

    def __init__(
        self,
        rng,
        n_in,
        n_per_base,
        n_out,
        n_layer=1,
        basefuncs1=None,
        basefuncs2=None,
        gradient=None,
        with_shortcuts=False,
    ):
        super().__init__()

        self.rng = rng
        self.n_layers = n_layer
        self.hidden_layers = nn.ModuleList()
        self.n_in = n_in
        self.n_out = n_out
        self.with_shortcuts = with_shortcuts
        self.fixL0 = False
        self.gradient = gradient

        for l in range(n_layer):
            if l == 0:
                n_input = n_in
            else:
                n_input = self.hidden_layers[l - 1].n_out

            hiddenLayer = FGLayer(
                rng=rng,
                n_in=n_input,
                n_per_base=n_per_base,
                basefuncs1=basefuncs1,
                basefuncs2=basefuncs2,
                layer_idx=l,
            )
            self.hidden_layers.append(hiddenLayer)

        if self.with_shortcuts:
            output_layer_n_in = sum([l.n_out for l in self.hidden_layers])
        else:
            output_layer_n_in = self.hidden_layers[-1].n_out

        self.output_layer = LinearRegression(
            rng=rng,
            n_in=output_layer_n_in,
            n_out=n_out,
        )

        self.optimizer = None
        self.extrapol_optimizer = None

    @staticmethod
    def vec_norm(vec):
        return torch.sqrt(torch.sum(vec ** 2))

    @staticmethod
    def vec_normalize(vec):
        norm = MLFG.vec_norm(vec)
        return vec / (norm + 1e-10)

    def forward_hidden(self, x):
        hidden_outputs = []

        for layer in self.hidden_layers:
            x = layer(x)
            hidden_outputs.append(x)

        if self.with_shortcuts:
            output_layer_inp = torch.cat([l for l in reversed(hidden_outputs)], dim=1)
        else:
            output_layer_inp = hidden_outputs[-1]

        return hidden_outputs, output_layer_inp

    def forward(self, x, div_thresh=1e-4):
        _, output_layer_inp = self.forward_hidden(x)
        output, denominator = self.output_layer(output_layer_inp, div_thresh=div_thresh)
        return output, denominator

    def get_params(self):
        params = []
        for layer in self.hidden_layers:
            params.extend(layer.get_params())
        params.extend(self.output_layer.get_params())
        return params

    def get_state(self):
        return [l.get_state() for l in self.hidden_layers] + [self.output_layer.get_state()]

    def set_state(self, newState):
        for s, l in zip(newState, list(self.hidden_layers) + [self.output_layer]):
            l.set_state(s)

    def evaluate(self, input):
        x = torch.tensor(cast_to_floatX(input), dtype=torch.float32)
        with torch.no_grad():
            output, _ = self.forward(x, div_thresh=0.0001)
        return output.detach().cpu().numpy()

    def get_n_units_type1(self):
        return sum([l.get_n_type1() for l in self.hidden_layers])

    def get_n_units_type2(self):
        return sum([l.get_n_type2() for l in self.hidden_layers])

    def L1_value(self):
        return self.output_layer.L1_value() + sum([l.L1_value() for l in self.hidden_layers])

    def L2_sqr_value(self):
        return self.output_layer.L2_sqr_value() + sum([l.L2_sqr_value() for l in self.hidden_layers])

    def get_num_active_units(self, thresh=0.1):
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

            for i in range(layer.get_n_type2()):
                if (
                    in_weight_norm[layer.get_n_type1() + i] > thresh
                    and in_weight_norm[layer.get_n_type1() + layer.get_n_type2() + i] > thresh
                ):
                    in_weight_norm[layer.get_n_type1() + i] += in_weight_norm[
                        layer.get_n_type1() + layer.get_n_type2() + i
                    ]
                else:
                    in_weight_norm[layer.get_n_type1() + i] = 0

            for i in range(layer.get_n_type1()):
                if (
                    out_weight_norm[i] * in_weight_norm[i] > thresh * thresh
                    and layer.get_nodes_type1()[i] != 0
                ):
                    total += 1

            for i in range(layer.get_n_type1(), layer.get_n_type1() + layer.get_n_type2()):
                if out_weight_norm[i] * in_weight_norm[i] > thresh * thresh:
                    total += 1

        return total

    def _make_optimizer(self, learning_rate):
        gradient = self.gradient

        if gradient == 'sgd+' or gradient == 'sgd' or gradient is None:
            return optim.SGD(self.parameters(), lr=learning_rate)
        elif gradient == 'adam':
            return optim.Adam(self.parameters(), lr=learning_rate, eps=1e-4)
        elif gradient == 'adadelta':
            return optim.Adadelta(self.parameters(), lr=learning_rate)
        elif gradient == 'rmsprop':
            return optim.RMSprop(self.parameters(), lr=learning_rate)
        elif gradient == 'nag':
            return optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

        raise ValueError("unknown gradient " + str(gradient))

    def _ensure_optimizers(self, learning_rate):
        if self.optimizer is None:
            self.optimizer = self._make_optimizer(learning_rate)
        if self.extrapol_optimizer is None:
            self.extrapol_optimizer = optim.Adam(self.parameters(), lr=learning_rate, eps=1e-4)

    def _float_to_sympy(self, x):
        return sp.Float(float(x), 12)

    def _sparse_affine_expr(self, inputs, weights, bias, thresh):
        expr = self._float_to_sympy(bias)

        for w, inp in zip(weights, inputs):
            if abs(w) > thresh:
                expr += self._float_to_sympy(w) * inp

        return expr

    def _build_symbolic_hidden(self, thresh):
        layer_inputs = [sp.Symbol(f'x{i}') for i in range(1, self.n_in + 1)]
        hidden_exprs = []

        for layer in self.hidden_layers:
            W = layer.get_params()[0]
            b = layer.get_params()[1]

            n_type1 = layer.get_n_type1()
            n_type2 = layer.get_n_type2()

            z_exprs = []
            for j in range(n_type1):
                z_exprs.append(self._sparse_affine_expr(layer_inputs, W[:, j], b[j], thresh))

            z1_exprs = []
            for j in range(n_type2):
                col = n_type1 + j
                z1_exprs.append(self._sparse_affine_expr(layer_inputs, W[:, col], b[col], thresh))

            z2_exprs = []
            for j in range(n_type2):
                col = n_type1 + n_type2 + j
                z2_exprs.append(self._sparse_affine_expr(layer_inputs, W[:, col], b[col], thresh))

            out_exprs = []

            for j, typ in enumerate(layer.get_nodes_type1()):
                expr = z_exprs[j]
                if typ == 0:
                    out_exprs.append(expr)
                elif typ == 1:
                    out_exprs.append(sp.sin(expr))
                elif typ == 2:
                    out_exprs.append(sp.cos(expr))
                else:
                    raise ValueError(f"Unknown unary node type {typ}")

            for j, typ in enumerate(layer.get_nodes_type2()):
                if typ == 0:
                    out_exprs.append(z1_exprs[j] * z2_exprs[j])
                else:
                    raise ValueError(f"Unknown binary node type {typ}")

            hidden_exprs.append(out_exprs)
            layer_inputs = out_exprs

        return hidden_exprs

    def get_symbolic_expression(self, thresh=1e-3, simplify_expr=True):
        hidden_exprs = self._build_symbolic_hidden(thresh)

        if self.with_shortcuts:
            output_inputs = []
            for layer_exprs in reversed(hidden_exprs):
                output_inputs.extend(layer_exprs)
        else:
            output_inputs = hidden_exprs[-1]

        W = self.output_layer.get_params()[0]
        b = self.output_layer.get_params()[1]

        expressions = []

        for out_idx in range(self.n_out):
            numerator = self._sparse_affine_expr(
                output_inputs,
                W[:, out_idx],
                b[out_idx],
                thresh,
            )
            denominator = self._sparse_affine_expr(
                output_inputs,
                W[:, self.n_out + out_idx],
                b[self.n_out + out_idx],
                thresh,
            )

            expr = numerator / denominator

            if simplify_expr:
                expr = sp.simplify(sp.expand(expr))

            expressions.append(expr)

        if self.n_out == 1:
            return expressions[0]

        return expressions

    def print_symbolic_expression(self, thresh=1e-3, simplify_expr=True):
        expr = self.get_symbolic_expression(thresh=thresh, simplify_expr=simplify_expr)
        print(expr)
        return expr

    def train_step(self, input, labels, L1_reg, L2_reg, fixL0, learning_rate, div_thresh):
        self.train()
        self._ensure_optimizers(learning_rate)

        x = torch.tensor(input, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)

        self.optimizer.zero_grad()

        output, denominator = self.forward(x, div_thresh=div_thresh)

        mse = self.output_layer.loss(output, y)
        l1 = self.L1_value()
        l2 = self.L2_sqr_value()
        penalty = self.output_layer.penalty_value(denominator, div_thresh)
        cost = mse + L1_reg * l1 + L2_reg * l2 + penalty

        cost.backward()

        if self.gradient == 'sgd+' or self.gradient == 'sgd' or self.gradient is None:
            for param in self.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1.0, 1.0)

        self.optimizer.step()

        if fixL0:
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if name.endswith(".W"):
                        mask = torch.abs(param.data) < 0.001
                        param.data[mask] = 0.0

        return float(cost.detach().cpu().item())

    def remove_extrapol_error_step(self, input, learning_rate, div_thresh):
        self.train()
        self._ensure_optimizers(learning_rate)

        x = torch.tensor(input, dtype=torch.float32)

        self.extrapol_optimizer.zero_grad()

        output, denominator = self.forward(x, div_thresh=div_thresh)
        extrapol_cost = self.output_layer.extrapol_loss_value(output, denominator, div_thresh)

        extrapol_cost.backward()
        self.extrapol_optimizer.step()

        return float(extrapol_cost.detach().cpu().item())

    def test_model(self, input, labels, div_thresh=0.0001):
        self.eval()

        x = torch.tensor(input, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)

        with torch.no_grad():
            output, _ = self.forward(x, div_thresh=div_thresh)
            return float(self.output_layer.loss(output, y).detach().cpu().item())

    def validate_model(self, input, labels, div_thresh=0.0001):
        return self.test_model(input, labels, div_thresh=div_thresh)

    def L1_loss(self):
        self.eval()
        with torch.no_grad():
            return self.L1_value().detach().cpu().numpy()

    def MSE(self, input, labels, div_thresh=0.0001):
        return self.test_model(input, labels, div_thresh=div_thresh)


def test_mlfg(
    datasets,
    learning_rate=0.01,
    L1_reg=0.001,
    L2_reg=0.00,
    n_epochs=200,
    batch_size=20,
    n_layer=1,
    n_per_base=5,
    basefuncs1=None,
    basefuncs2=None,
    with_shortcuts=False,
    id=None,
    classifier=None,
    gradient=None,
    init_state=None,
    verbose=True,
    param_store=None,
    reg_start=0,
    reg_end=None,
    validate_every=10,
    k=100,
):
    train_set_x, train_set_y = cast_dataset_to_floatX(datasets[0])
    valid_set_x, valid_set_y = cast_dataset_to_floatX(datasets[1])
    MAX_INPUT_VAL = np.max(abs(train_set_x))
    print("Max input value is: ", MAX_INPUT_VAL)

    if len(datasets) > 2 and len(datasets[2]) == 2:
        test_set_x, test_set_y = cast_dataset_to_floatX(datasets[2])
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
        classifier = MLFG(
            rng=rng,
            n_in=inputdim,
            n_per_base=n_per_base,
            n_out=outputdim,
            n_layer=n_layer,
            gradient=gradient,
            basefuncs1=basefuncs1,
            basefuncs2=basefuncs2,
            with_shortcuts=with_shortcuts,
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
    extrapol_train_errors = []
    validation_errors = []
    test_errors = []
    MSE = []
    L1 = []

    if param_store is not None:
        param_store.append(classifier.get_params())

    if reg_end is None:
        reg_end = 0

    while (epoch < n_epochs) and (not done_looping):
        special_penalty = 0
        epoch = epoch + 1
        reg_factor = 0.0

        if reg_start < epoch <= reg_end:
            reg_factor = 1.0
            L1.append([epoch, float(np.asarray(classifier.L1_loss()).item())])
            if (epoch - reg_start) % k == 0 and epoch < reg_end:
                special_penalty = 1

        temp = list(zip(list(train_set_x), list(train_set_y)))
        random.shuffle(temp)
        train_set_x, train_set_y = list(zip(*temp))
        train_set_x = np.asarray(train_set_x)
        train_set_y = np.asarray(train_set_y)

        minibatch_avg_cost = 0.0
        for minibatch_index in range(n_train_batches):
            index = minibatch_index
            minibatch_avg_cost += classifier.train_step(
                input=train_set_x[index * batch_size: (index + 1) * batch_size],
                labels=train_set_y[index * batch_size: (index + 1) * batch_size],
                L1_reg=L1_reg * reg_factor,
                L2_reg=L2_reg * reg_factor,
                fixL0=epoch > reg_end,
                div_thresh=1.0 / np.sqrt(epoch + 1),
                learning_rate=learning_rate,
            )

        if special_penalty == 1:
            n_num, n_in = train_set_x.shape
            extra_set_x = (2 * np.random.rand(n_num, n_in) - 1.0) * MAX_INPUT_VAL
            assert extra_set_x.shape == train_set_x.shape

            for x in range(n_num):
                for y in range(n_in):
                    if extra_set_x[x][y] >= 0.0:
                        extra_set_x[x][y] += MAX_INPUT_VAL
                    else:
                        extra_set_x[x][y] -= MAX_INPUT_VAL

            extrapol_error_training = 0.0

            for minibatch_index in range(n_train_batches):
                index = minibatch_index
                extrapol_error_training += classifier.remove_extrapol_error_step(
                    input=extra_set_x[index * batch_size: (index + 1) * batch_size],
                    div_thresh=1.0 / np.sqrt(epoch + 1),
                    learning_rate=learning_rate,
                )

            extrapol_train_errors.append([epoch, extrapol_error_training / n_train_batches])

        train_errors.append([epoch, minibatch_avg_cost / n_train_batches])

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

            this_MSE = [
                classifier.MSE(
                    input=train_set_x[index * batch_size:(index + 1) * batch_size],
                    labels=train_set_y[index * batch_size:(index + 1) * batch_size],
                )
                for index in range(n_train_batches)
            ]
            MSE.append([epoch, float(np.mean(this_MSE))])

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
                        'epoch %i, minibatch %i/%i, test error of best model %f'
                        % (epoch, minibatch_index + 1, n_train_batches, test_score)
                    )

        if epoch % 500 == 0:
            print(
                "Epoch: ",
                epoch,
                "\tBest val error: ",
                best_validation_error,
                "\tcurrent val error: ",
                this_validation_error,
            )
            sys.stdout.flush()

    end_time = timeit.default_timer()
    time_required = (end_time - start_time) / 60.0

    print(
        'Optimization complete. Best validation score of %f obtained at epoch %i, with test performance %f '
        % (best_validation_error, best_epoch, test_score)
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
        'extrapol_train_losses': np.asarray(extrapol_train_errors),
        'MSE': np.asarray(MSE),
        'L1': np.asarray(L1),
        'val_errors': np.asarray(validation_errors),
        'test_errors': np.asarray(test_errors),
        'classifier': classifier,
        'test_score': test_score,
        'val_score': this_validation_error,
        'best_val_score': best_validation_error,
        'best_epoch': best_epoch,
        'best_state': best_state,
        'num_active': classifier.get_num_active_units(),
        'runtime': time_required,
    }


def usage():
    print(
        sys.argv[0]
        + "[-i id -d dataset -p extrapolationdataset -l layers -e epochs -n nodes -r learningrate --initfile=file --batchsize=k --l1=l1reg --l2=l2reg --shortcut --reg_start=start --reg_end=end --resfolder -v]"
    )


if __name__ == "__main__":
    dataset_file = None
    extra_pol_test_sets = []
    extra_pols = []
    n_epochs = 1200
    n_layers = 3
    n_nodes = 5
    batch_size = 20
    init_file = None
    init_state = None
    gradient = "sgd"
    L1_reg = 0.001
    L2_reg = 0.001
    learning_rate = 0.01
    with_shortcuts = False
    reg_start = 0
    reg_end = None
    output = False
    verbose = 0
    k = 99999999999
    id = np.random.randint(0, 1000000)
    result_folder = "./"
    basefuncs1 = [0, 1, 2]
    iterNum = 0

    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hv:i:d:p:l:e:n:f:co",
            [
                "help", "verbose=", "id=", "dataset=", "extrapol=", "layers=", "epochs=",
                "nodes=", "l1=", "l2=", "lr=", "resfolder=", "batchsize=", "initfile=",
                "gradient=", "reg_start=", "reg_end=", "shortcut", "output", "k_update=", "iterNum="
            ],
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
        elif opt in ("-c", "--shortcut"):
            with_shortcuts = True
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
        elif opt in ("--iterNum"):
            iterNum = int(arg)
        elif opt in ("--k_update"):
            k = int(arg)

    if not dataset_file:
        print("provide datasetfile!")
        usage()
        exit(1)

    dataset = load_data(dataset_file)

    if len(extra_pol_test_sets) > 0:
        if verbose > 0:
            print("do also extrapolation test(s)!")
        extra_pols = [load_data(test_set) for test_set in extra_pol_test_sets]

    if init_file:
        with open(init_file, 'rb') as f:
            init_state = pickle.load(f, encoding='latin1')
            print("load initial state from file " + init_file)

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    name = result_folder + "/" + str(id)
    print("Results go into " + result_folder)

    result = test_mlfg(
        datasets=dataset,
        k=k,
        n_epochs=n_epochs,
        verbose=verbose > 0,
        learning_rate=learning_rate,
        L1_reg=L1_reg,
        L2_reg=L2_reg,
        basefuncs2=[0],
        basefuncs1=basefuncs1,
        n_layer=n_layers,
        n_per_base=n_nodes,
        id=id,
        gradient=gradient,
        batch_size=batch_size,
        init_state=init_state,
        reg_start=reg_start,
        reg_end=reg_end,
        with_shortcuts=with_shortcuts,
    )

    classifier = result['classifier']

    with open(name + '.best_state', 'wb') as f:
        pickle.dump(result['best_state'], f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(name + '.last_state', 'wb') as f:
        pickle.dump(classifier.get_state(), f, protocol=pickle.HIGHEST_PROTOCOL)

    extra_scores = []
    extra_scores_best = []

    for extra in extra_pols:
        extra_set_x, extra_set_y = cast_dataset_to_floatX(extra[0])
        extra_scores.append(classifier.test_model(input=extra_set_x, labels=extra_set_y))

    classifier.set_state(result['best_state'])

    for extra in extra_pols:
        extra_set_x, extra_set_y = cast_dataset_to_floatX(extra[0])
        extra_scores_best.append(classifier.test_model(input=extra_set_x, labels=extra_set_y))

    result_line = ""
    with open(name + '.res', 'w') as f:
        if id <= 0:
            f.write(
                '#C k iter layers epochs nodes lr L1 L2 shortcut batchsize regstart regend'
                + ' id dataset gradient numactive bestnumactive bestepoch runtime'
                + "".join([' extrapol' + str(i) for i in range(1, len(extra_scores) + 1)])
                + "".join([' extrapolbest' + str(i) for i in range(1, len(extra_scores_best) + 1)])
                + ' valerror valerrorbest testerror\n'
            )
            f.write('# extra datasets: ' + " ".join(extra_pol_test_sets) + '\n')

        result_line = [
            str(k),
            str(iterNum),
            str(n_layers),
            str(n_epochs),
            str(n_nodes),
            str(learning_rate),
            str(L1_reg),
            str(L2_reg),
            str(with_shortcuts),
            str(batch_size),
            str(reg_start),
            str(reg_end),
            str(id),
            dataset_file,
            gradient,
            str(result['num_active']),
            str(classifier.get_num_active_units()),
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

    with open(name + '.MSE', 'w', newline='') as csvfile:
        a = csv.writer(csvfile, delimiter='\t')
        a.writerows([["#C epoch", "MSE"]])
        a.writerows([["# "] + result_line])
        a.writerows(result['MSE'])

    with open(name + '.L1', 'w', newline='') as csvfile:
        a = csv.writer(csvfile, delimiter='\t')
        a.writerows([["#C epoch", "L1"]])
        a.writerows([["# "] + result_line])
        a.writerows(result['L1'])

    output = 1
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

        with open(name + '.extrapoltrainloss', 'w', newline='') as csvfile:
            a = csv.writer(csvfile, delimiter='\t')
            a.writerows([["#C epoch", "extrapol_train_loss"]])
            a.writerows([["# "] + result_line])
            a.writerows(result['extrapol_train_losses'])