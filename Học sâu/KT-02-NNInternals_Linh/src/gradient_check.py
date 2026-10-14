import numpy as np

def kiem_chung_gradient(f, x, grad_giai_tich, eps=1e-5, verbose=True, ten=""):
    grad_so = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        goc = x[i]
        x[i] = goc + eps; f_cong = f()
        x[i] = goc - eps; f_tru = f()
        x[i] = goc
        grad_so[i] = (f_cong - f_tru) / (2 * eps)
    sai_lech = np.abs(grad_so - grad_giai_tich).max()
    if verbose:
        print(f"  {ten:<6} sai lệch lớn nhất: {sai_lech:.2e} → {'ĐÚNG' if sai_lech < 1e-7 else 'SAI'}")
    return sai_lech


def check_network(net, X, y, eps=1e-5, tol=1e-7, verbose=True):
    net.forward(X, y)
    grads = net.backward()
    worst = 0.0
    for name, g in grads.items():
        err = kiem_chung_gradient(lambda: net.forward(X, y), net.p[name], g, eps, verbose, name)
        worst = max(worst, err)
    net.forward(X, y)
    if verbose:
        print(f"=> Sai lệch lớn nhất toàn mạng: {worst:.2e} → {'ĐÚNG' if worst < tol else 'SAI'}")
    return worst

print("OK")