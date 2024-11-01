from torch import nn
import torch
import torch.optim as optim

import numpy as np
from tqdm.auto import tqdm
from models.testing_models.s4d import S4D, dropout_fn


class S4Model(nn.Module):

    def __init__(
            self,
            d_input,
            d_state,
            d_output=10,
            d_model=256,
            n_layers=4,
            dropout=0.2,
            prenorm=False,
            bidirectional=False,
            seq2seq=False,
            lr=0.001,
            activation=nn.Identity()
    ):
        super().__init__()

        self.prenorm = prenorm
        self.seq2seq = seq2seq

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(
                    d_model=d_model,
                    d_state=d_state,
                    bidirectional=bidirectional,
                    # postact='glu' if glu else None,
                    dropout=dropout,
                    transposed=True,
                    lr=min(0.001, lr)
                )
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout) if dropout > 0 else nn.Identity())

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)
        self.act = activation

    def forward(self, x, aux=None, t=None, **kwargs):
        """
        Input x is shape (B, L, d_input)
        """

        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        if not self.seq2seq:
            x = x.mean(1)
        # Decode the outputs
        x = self.decoder(x)  # (B, L, d_model) -> (B, L, d_output)
        x = self.act(x)
        return x, None

    def default_state(self, *args, **kwargs):
        return [layer.default_state(*args, **kwargs) for layer in self.s4_layers]


class Loss(nn.Module):
    def __init__(self, name, reg=1.0, transform=lambda x: x, threshold=10., backward=False, norm_foo=lambda x: x):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()

    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)


def histogram_torch(x, n_bins, density=True):
    a, b = x.min().item(), x.max().item()
    delta = (b - a) / n_bins
    bins = torch.arange(a, b + 1e-8, step=delta)
    count = torch.histc(x, n_bins).float()
    if density:
        count = count / delta / float(x.shape[0] * x.shape[1])
    return count, bins


class HistoLoss(Loss):
    def __init__(self, x_real, n_bins, **kwargs):
        super(HistoLoss, self).__init__(**kwargs)
        self.densities = list()
        self.locs = list()
        self.deltas = list()
        for i in range(x_real.shape[2]):
            x_i = x_real[..., i].reshape(-1, 1)
            d, b = histogram_torch(x_i, n_bins, density=True)
            self.densities.append(nn.Parameter(d).to(x_real.device))
            delta = b[1:2] - b[:1]
            loc = 0.5 * (b[1:] + b[:-1])
            self.locs.append(loc)
            self.deltas.append(delta)

    def compute(self, x_fake):
        loss = list()

        def relu(x):
            return x * (x >= 0.).float()

        for i in range(x_fake.shape[2]):
            loc = self.locs[i].view(1, -1).to(x_fake.device)
            x_i = x_fake[:, :, i].contiguous().view(-1, 1).repeat(1, loc.shape[1])
            dist = torch.abs(x_i - loc)
            counter = (relu(self.deltas[i].to(x_fake.device) / 2. - dist) > 0.).float()
            density = counter.mean(0) / self.deltas[i].to(x_fake.device)
            abs_metric = torch.abs(density - self.densities[i].to(x_fake.device))
            loss.append(torch.mean(abs_metric, 0))
        loss_componentwise = torch.stack(loss)
        return loss_componentwise


def compute_classification_score(x_fake, x_real, get_optim_func, device):
    x_fake = x_fake.detach().cpu()
    x_real = x_real.detach().cpu()
    X = torch.cat([x_fake, x_real], dim=0)
    Y = torch.cat([torch.ones_like(x_fake[:, 0, 0]), torch.zeros_like(x_real[:, 0, 0])], dim=0)

    randperm = torch.randperm(X.shape[0])

    X_train, Y_train = X[randperm[:int(X.shape[0] * 0.8)]], Y[randperm[:int(X.shape[0] * 0.8)]]

    X_test, Y_test = X[randperm[int(X.shape[0] * 0.8):]], Y[randperm[int(X.shape[0] * 0.8):]]
    model = S4Model(d_input=X.shape[-1], d_state=16, d_output=1, d_model=16, n_layers=1,
                    dropout=0.0, seq2seq=False).to(device)
    trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, Y_train), shuffle=True,
                                              batch_size=128)
    testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, Y_test), batch_size=128)
    optimizer, _ = get_optim_func(model, lr=0.01, weight_decay=0.0, epochs=100)

    pbar = tqdm(range(100))
    for i in range(100):
        for data, label in trainloader:
            optimizer.zero_grad()
            pred, _ = model(data.to(device))

            loss = torch.nn.BCEWithLogitsLoss()(pred.squeeze(-1), label.to(device))
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            test_loss = 0
            for ind, (data, label) in enumerate(testloader):
                pred, _ = model(data.to(device))
                loss = torch.nn.BCEWithLogitsLoss()(pred.squeeze(-1), label.to(device)).detach().cpu()
                test_loss += loss

            pbar.set_description(f'Epoch {i} Test loss: {test_loss / (ind + 1)}')

    return test_loss


def compute_predictive_score(x_real, x_fake, pred_step, get_optim_func, device, pred_activation):
    x_fake = x_fake.detach().cpu()
    x_real = x_real.detach().cpu()
    X = x_fake[:, :-1]
    Y = x_fake[:, 1:]
    masks = torch.ones_like(X, dtype=torch.bool)
    masks[:, :-pred_step] = 0
    X_test = x_real[:, :-1]
    Y_test = x_real[:, 1:]

    model = S4Model(d_input=X.shape[-1], d_state=16, d_output=1, d_model=16, n_layers=1,
                    dropout=0.0, seq2seq=True, activation=pred_activation).to(device)
    trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y, masks), shuffle=True, batch_size=128)
    testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, Y_test, masks), batch_size=128)
    optimizer, _ = get_optim_func(model, lr=0.01, weight_decay=0.0, epochs=100)

    pbar = tqdm(range(100))
    for i in range(100):
        for data, target, mask in trainloader:
            mask = mask.to(device)
            optimizer.zero_grad()
            pred, _ = model(data.to(device))
            loss = torch.nn.MSELoss()(pred[mask], target.to(device)[mask])
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            test_loss = 0
            for ind, (data, target, mask) in enumerate(testloader):
                pred, _ = model(data.to(device))
                loss = torch.nn.MSELoss()(pred[mask], target.to(device)[mask]).detach().cpu()
                test_loss += loss

            pbar.set_description(f'Epoch {i} Test loss: {test_loss / (ind + 1)}')
    return test_loss


def compute_test_metrics(x_fake, x_real):
    res = dict()
    res['marginal_loss'] = HistoLoss(x_real=x_real, n_bins=50, name='marginal_loss')(x_fake).item()

    return res


def compute_all_metrics(x_real, gens, get_optim_func, pred_activation, device):
    pred_step = 10
    cls = []
    pred = []
    marg = []
    for i in range(10):
        clfscore = compute_classification_score(gens, x_real, get_optim_func, device)
        predscore = compute_predictive_score(x_real, gens, pred_step, get_optim_func, device, pred_activation)
        marginalscore = compute_test_metrics(gens, x_real)['marginal_loss']
        cls.append(clfscore)
        pred.append(predscore)
        marg.append(marginalscore)
    clfscore, clf_std = np.mean(cls), np.std(cls)
    predscore, pred_std = np.mean(pred), np.std(pred)
    marginalscore = np.mean(marg)

    res = {'clf_score_mean': clfscore,
           'clf_score_std': clf_std,
           'marginal_score_mean': marginalscore,
           'predictive_score_mean': predscore,
           'predictive_score_std': pred_std
           }

    return res


def setup_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.
    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.
    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
                             f"Optimizer group {i}",
                             f"{len(g['params'])} tensors",
                         ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler
