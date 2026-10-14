import os
import time
import numpy as np
from scipy.special import erf
from sklearn.datasets import make_moons, load_digits
from sklearn.model_selection import train_test_split

# ======================================================================
# 1. HÀM KÍCH HOẠT
# ======================================================================
# 7 hàm kích hoạt + đạo hàm (NumPy thuần).

def sigmoid(z):
    z = np.clip(z, -500, 500)               # tránh tràn số exp
    return 1.0 / (1.0 + np.exp(-z))

def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)                      # max = 0.25 tại z=0

def tanh(z):
    return np.tanh(z)

def d_tanh(z):
    return 1 - np.tanh(z) ** 2              # max = 1 tại z=0

def relu(z):
    return np.maximum(0.0, z)

def d_relu(z):
    return (z > 0).astype(z.dtype)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def d_leaky_relu(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)

def elu(z, alpha=1.0):
    return np.where(z > 0, z, alpha * (np.exp(np.minimum(z, 0)) - 1))

def d_elu(z, alpha=1.0):
    return np.where(z > 0, 1.0, alpha * np.exp(np.minimum(z, 0)))

def _Phi(z):                                # CDF chuẩn
    return 0.5 * (1 + erf(z / np.sqrt(2)))

def _phi(z):                                # PDF chuẩn
    return np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)

def gelu(z):
    return z * _Phi(z)

def d_gelu(z):
    return _Phi(z) + z * _phi(z)

def silu(z):
    return z * sigmoid(z)

def d_silu(z):
    s = sigmoid(z)
    return s * (1 + z * (1 - s))

ACTIVATIONS = {
    "sigmoid": (sigmoid, d_sigmoid),
    "tanh": (tanh, d_tanh),
    "relu": (relu, d_relu),
    "leaky_relu": (leaky_relu, d_leaky_relu),
    "elu": (elu, d_elu),
    "gelu": (gelu, d_gelu),
    "silu": (silu, d_silu),
}

PRETTY = {
    "sigmoid": "Sigmoid", "tanh": "Tanh", "relu": "ReLU",
    "leaky_relu": "LeakyReLU", "elu": "ELU", "gelu": "GELU", "silu": "SiLU/Swish",
}

# ======================================================================
# 2. OPTIMIZER
# ======================================================================
# Optimizer tự cài: SGD (có momentum) và Adam. Cập nhật tại chỗ trên dict tham số.

class SGD:
    def __init__(self, lr=0.1, momentum=0.0):
        self.lr, self.mu, self.v = lr, momentum, {}

    def step(self, params, grads):
        for k, g in grads.items():
            if self.mu:
                self.v[k] = self.mu * self.v.get(k, 0.0) - self.lr * g
                params[k] += self.v[k]
            else:
                params[k] -= self.lr * g

class Adam:

    def __init__(self, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, params, grads):
        self.t += 1
        for k, g in grads.items():
            m = self.m.get(k, np.zeros_like(g))
            v = self.v.get(k, np.zeros_like(g))
            m = self.b1 * m + (1 - self.b1) * g
            v = self.b2 * v + (1 - self.b2) * g * g
            self.m[k], self.v[k] = m, v
            m_hat = m / (1 - self.b1 ** self.t)      # hiệu chỉnh lệch khởi đầu
            v_hat = v / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# ======================================================================
# 3. PHẦN 1 — XOR
# ======================================================================
# PHẦN 1 — mạng 2→4→1 giải XOR bằng NumPy thuần (tự viết forward + backward).

_sig = lambda z: 1 / (1 + np.exp(-z))

X_XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
Y_XOR = np.array([[0], [1], [1], [0]], dtype=np.float64)       # shape (4,1) — rất quan trọng!

class MLP:

    def __init__(self, n_in=2, n_hid=4, n_out=1, seed=0, scale=1.0):
        r = np.random.RandomState(seed)
        self.p = {
            "W1": r.randn(n_in, n_hid) * scale, "b1": np.zeros(n_hid),
            "W2": r.randn(n_hid, n_out) * scale, "b2": np.zeros(n_out),
        }

    def forward(self, X, y):
        p = self.p
        self.X, self.y = X, y
        self.a1 = _sig(X @ p["W1"] + p["b1"])
        self.a2 = _sig(self.a1 @ p["W2"] + p["b2"])
        return np.mean((self.a2 - y) ** 2)

    def backward(self):
        p, N = self.p, len(self.y)                       # nhớ chia N
        d_a2 = 2 * (self.a2 - self.y) / N                # ∂L/∂a2
        d_z2 = d_a2 * self.a2 * (1 - self.a2)            # qua sigmoid: σ(1−σ)
        d_a1 = d_z2 @ p["W2"].T                          # sai số "chảy ngược"
        d_z1 = d_a1 * self.a1 * (1 - self.a1)
        return {"W1": self.X.T @ d_z1, "b1": d_z1.sum(0),
                "W2": self.a1.T @ d_z2, "b2": d_z2.sum(0)}

    def predict(self, X):
        p = self.p
        return _sig(_sig(X @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"])

class LogisticNoHidden:

    def __init__(self, n_in=2, n_out=1, seed=0):
        r = np.random.RandomState(seed)
        self.p = {"W": r.randn(n_in, n_out), "b": np.zeros(n_out)}

    def forward(self, X, y):
        self.X, self.y = X, y
        self.a = _sig(X @ self.p["W"] + self.p["b"])
        return np.mean((self.a - y) ** 2)

    def backward(self):
        N = len(self.y)
        d_z = 2 * (self.a - self.y) / N * self.a * (1 - self.a)
        return {"W": self.X.T @ d_z, "b": d_z.sum(0)}

    def predict(self, X):
        return _sig(X @ self.p["W"] + self.p["b"])

def train(net, X, y, epochs=10000, lr=1.0, opt=None):
    """Full-batch. Trả về list loss theo epoch."""
    opt = opt or SGD(lr)
    losses = []
    for _ in range(epochs):
        losses.append(net.forward(X, y))
        opt.step(net.p, net.backward())
    return np.array(losses)

# ======================================================================
# 4. MẠNG SÂU
# ======================================================================
# Mạng nhiều tầng tổng quát bằng NumPy: 7 activation, 5 cách khởi tạo, BatchNorm tuỳ chọn.

INITS = ["zeros", "const", "small", "xavier", "he"]


def init_weights(n_in, n_out, kind, rng, const=0.1, small=0.01):
    if kind == "zeros":
        return np.zeros((n_in, n_out))
    if kind == "const":                               # mọi neuron cùng giá trị (≠ 0)
        return np.full((n_in, n_out), const)
    if kind == "small":                               # random nhỏ N(0, 0.01²)
        return rng.randn(n_in, n_out) * small
    if kind == "xavier":                              # Var(W) = 2/(n_in+n_out)  — cho tanh/sigmoid
        return rng.randn(n_in, n_out) * np.sqrt(2.0 / (n_in + n_out))
    if kind == "he":                                  # Var(W) = 2/n_in          — cho ReLU
        return rng.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
    raise ValueError(kind)


class DeepNet:
    def __init__(self, sizes, act="relu", init="he", batchnorm=False, seed=0, bn_eps=1e-5, bn_mom=0.1):
        self.sizes, self.L, self.act_name = list(sizes), len(sizes) - 1, act
        self.f, self.df = ACTIVATIONS[act]
        self.bn, self.bn_eps, self.bn_mom = batchnorm, bn_eps, bn_mom
        rng = np.random.RandomState(seed)
        self.p, self.run_mean, self.run_var = {}, {}, {}
        for i in range(self.L):
            self.p[f"W{i}"] = init_weights(sizes[i], sizes[i + 1], init, rng)
            self.p[f"b{i}"] = np.zeros(sizes[i + 1])
            if batchnorm and i < self.L - 1:
                self.p[f"g{i}"] = np.ones(sizes[i + 1])      # gamma
                self.p[f"be{i}"] = np.zeros(sizes[i + 1])    # beta
                self.run_mean[i] = np.zeros(sizes[i + 1])
                self.run_var[i] = np.ones(sizes[i + 1])

    # ------------------------------------------------------------------ FORWARD
    def forward(self, X, y=None, train=True):
        p, a = self.p, X
        self.cache = []
        for i in range(self.L - 1):
            z = a @ p[f"W{i}"] + p[f"b{i}"]
            c = {"a_in": a, "z": z}
            if self.bn:
                if train:
                    mu, var = z.mean(0), z.var(0)
                    self.run_mean[i] = (1 - self.bn_mom) * self.run_mean[i] + self.bn_mom * mu
                    self.run_var[i] = (1 - self.bn_mom) * self.run_var[i] + self.bn_mom * var
                else:
                    mu, var = self.run_mean[i], self.run_var[i]
                std = np.sqrt(var + self.bn_eps)
                zh = (z - mu) / std
                h = p[f"g{i}"] * zh + p[f"be{i}"]
                c.update(zh=zh, std=std)
            else:
                h = z
            a = self.f(h)
            c.update(h=h, a=a)
            self.cache.append(c)
        self.a_last = a
        logits = a @ p[f"W{self.L - 1}"] + p[f"b{self.L - 1}"]
        logits = logits - logits.max(1, keepdims=True)
        e = np.exp(logits)
        self.probs = e / e.sum(1, keepdims=True)
        self.y = y
        if y is None:
            return self.probs
        return -np.mean(np.log(self.probs[np.arange(len(y)), y] + 1e-300))

    def predict(self, X):
        return self.forward(X, None, train=False).argmax(1)

    # ----------------------------------------------------------------- BACKWARD
    def backward(self):
        p, N, g = self.p, len(self.y), {}
        d = self.probs.copy()
        d[np.arange(N), self.y] -= 1
        d /= N                                                    # ∂L/∂logits (softmax+CE) — nhớ chia N
        last = self.L - 1
        a_in = self.cache[-1]["a"] if self.cache else None
        g[f"W{last}"], g[f"b{last}"] = a_in.T @ d, d.sum(0)
        da = d @ p[f"W{last}"].T
        for i in reversed(range(self.L - 1)):
            c = self.cache[i]
            dh = da * self.df(c["h"])                             # qua activation
            if self.bn:
                zh, std = c["zh"], c["std"]
                g[f"g{i}"], g[f"be{i}"] = (dh * zh).sum(0), dh.sum(0)
                dzh = dh * p[f"g{i}"]
                dz = (N * dzh - dzh.sum(0) - zh * (dzh * zh).sum(0)) / (N * std)   # backward BatchNorm
            else:
                dz = dh
            g[f"W{i}"], g[f"b{i}"] = c["a_in"].T @ dz, dz.sum(0)
            da = dz @ p[f"W{i}"].T                                # sai số chảy ngược về tầng trước
        return g

# ======================================================================
# 5. HUẤN LUYỆN & CHẨN ĐOÁN
# ======================================================================
# Vòng huấn luyện mini-batch dùng chung + hàm chẩn đoán (đếm neuron chết, thống kê gradient/activation).

def accuracy(net, X, y):
    return float((net.predict(X) == y).mean())

def fit(net, Xtr, ytr, Xva, yva, epochs=50, lr=0.1, batch=64, seed=0, opt=None, thr=0.95, drop_last=True):
    opt = opt or SGD(lr)
    rng = np.random.RandomState(seed)
    N = len(Xtr)
    h = {"loss": [], "train_acc": [], "val_acc": [], "diverged": False, "epoch_thr": None}
    t0 = time.time()
    with np.errstate(all="ignore"):
        for ep in range(epochs):
            idx = rng.permutation(N)
            stop = N - batch + 1 if (drop_last and N >= batch) else N
            for s in range(0, stop, batch):
                b = idx[s:s + batch]
                loss = net.forward(Xtr[b], ytr[b])
                if not np.isfinite(loss):
                    h["diverged"] = True
                    break
                opt.step(net.p, net.backward())
            if h["diverged"]:
                break
            l = net.forward(Xtr, ytr, train=False)
            h["loss"].append(float(l))
            h["train_acc"].append(accuracy(net, Xtr, ytr))
            h["val_acc"].append(accuracy(net, Xva, yva))
            if h["epoch_thr"] is None and h["val_acc"][-1] >= thr:
                h["epoch_thr"] = ep + 1
    h["time"] = time.time() - t0
    return h

def dead_neurons(net, X):
    with np.errstate(all="ignore"):
        net.forward(X, None, train=False)
    per_layer = [float((~(c["h"] > 0).any(axis=0)).mean()) for c in net.cache]
    n = [c["h"].shape[1] for c in net.cache]
    total = sum(f * k for f, k in zip(per_layer, n)) / sum(n)
    return total, per_layer

def layer_grad_norms(net, X, y):
    net.forward(X, y)
    g = net.backward()
    return np.array([np.linalg.norm(g[f"W{i}"]) for i in range(net.L)])

# ======================================================================
# 6. DỮ LIỆU
# ======================================================================
# Nạp dữ liệu. MNIST thật nếu có (đặt biến môi trường USE_MNIST=1 và có mạng/ổ cache),

def moons(n=1000, noise=0.2, seed=0):
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return X.astype(np.float64), y.astype(int)

def load_digits_data(seed=0):
    d = load_digits()
    X, y = d.data / 16.0, d.target
    return train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)

def load_mnist_data(n_train=60000):                  # CHƯA chạy thử trong môi trường nộp bài (không có mạng)
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X, y = X / 255.0, y.astype(int)
    return X[:n_train], X[60000:], y[:n_train], y[60000:]

def load_image_data():
    if os.environ.get("USE_MNIST") == "1":
        return (*load_mnist_data(), "MNIST")
    return (*load_digits_data(), "sklearn digits (8×8) — thay MNIST")


print("da check xong")