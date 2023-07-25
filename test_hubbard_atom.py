
import itertools
import numpy as np

# ----------------------------------------------------------------------

from triqs.gf import Gf
from triqs.gf import MeshImTime, MeshImFreq

from triqs.gf import GfImTime, GfImFreq
from triqs.operators import c, c_dag, n

from triqs.gf import inverse, iOmega_n, Fourier

from triqs_tprf.rpa_tensor import get_rpa_tensor
from triqs_tprf.OperatorUtils import quadratic_matrix_from_operator

# ----------------------------------------------------------------------

from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization

from fitdlr import pydlr_driver
from fitdlr import BlockSymmetrizer
from fitdlr import constrained_lstsq_dlr_from_tau

# ----------------------------------------------------------------------

def get_H(U=1., mu=0.0, B=0.05):
    H = U * (n('up', 0) - 0.5)*(n('dn', 0) - 0.5) - mu * (n('up', 0) + n('dn', 0)) + B * (n('up', 0) - n('dn', 0))
    return H


def get_gfs(H, beta, fundamental_operators, niw=128):

    ntau = 6 * niw + 1
    
    ed = TriqsExactDiagonalization(H, fundamental_operators, beta)

    tmesh = MeshImTime(beta, 'Fermion', ntau)
    wmesh = MeshImFreq(beta, 'Fermion', niw)

    G_tau = Gf(mesh=tmesh, target_shape=(2,2))
    G_iw = Gf(mesh=wmesh, target_shape=(2,2))

    s = ['up', 'dn']
    for i, j in itertools.product(range(2), repeat=2):
        si, sj = s[i], s[j]
        ed.set_g2_tau(G_tau[i, j], c(si, 0), c_dag(sj, 0))
        ed.set_g2_iwn(G_iw[i, j], c(si, 0), c_dag(sj, 0))

    return G_tau, G_iw
    
        
def test_fit(verbose=False):

    np.random.seed(seed=1337)
    
    xi = -1.
    beta = 20.0

    H0 = get_H(U=0.)
    H = get_H()
    H_int = H - H0

    fundamental_operators = [c('up', 0), c('dn', 0)]

    G0_tau, G0_iw = get_gfs(H0, beta, fundamental_operators)
    G_tau, G_iw = get_gfs(H, beta, fundamental_operators)

    Sigma_iw = G_iw.copy()
    Sigma_iw << inverse(G0_iw) - inverse(G_iw)
    
    h_ab = quadratic_matrix_from_operator(H, fundamental_operators)
    U_abcd = get_rpa_tensor(H, fundamental_operators)

    # -- Fit greens function with DLR expansion

    tol = 1e-4
    G_tau.data[:] += np.random.normal(scale=tol, size=G_tau.data.shape)

    G_iaa = G_tau.data
    tau_i = np.array([float(t) for t in G_tau.mesh])

    from pydlr import dlr
    d = dlr(lamb=20., eps=1e-10)
    dd = pydlr_driver(d, beta)
        
    block_mat = np.array([[1, 0], [0, 2]])
    sym = BlockSymmetrizer(len(d), block_mat)

    opt = dict(
        discontinuity=True,
        density=True,
        realvalued=True,
        ftol=1e-10,
        )
    
    G_xaa_sym, sol = constrained_lstsq_dlr_from_tau(
        dd, h_ab, U_abcd, tau_i, G_iaa, beta, symmetrizer=sym, **opt)

    if verbose:
        print(f'tau-fit: {sol.message}')
        print(f'tau-fit: nfev {sol.nfev} nit {sol.nit} njev {sol.njev} success {sol.success}')
        print(f'tau-fit: res (g, rho, norm) = ({sol.res:1.1E}, {sol.density_res:1.1E}, {sol.norm_res:1.1E})')

    assert( sol.success == True )
    assert( sol.res < 1e-3 )
    assert( sol.density_res < 1e-9 )
    assert( sol.norm_res < 1e-9 )
    
    if verbose:
    
        d = dd.dlr
        G_laa_sym = d.tau_from_dlr(G_xaa_sym)

        G_xaa = G_xaa_sym
        G_laa = G_laa_sym

        G_tau_fit = G_tau.copy()
        G_tau_fit.data[:] = d.eval_dlr_tau(G_xaa, tau_i, beta)
        G_tau_diff = np.max(np.abs(G_tau.data - G_tau_fit.data))
        print(f'G_tau_diff = {G_tau_diff:1.1E}')
        #np.testing.assert_array_almost_equal(G_tau.data, G_tau_fit.data, decimal=tol*0.1)

        iwn = np.array([complex(w) for w in G_iw.mesh])
        G_iw_fit = G_iw.copy()
        G_iw_fit.data[:] = d.eval_dlr_freq(G_xaa, iwn, beta)
        G_iw_diff = np.max(np.abs(G_iw.data - G_iw_fit.data))
        print(f'G_iw_diff = {G_iw_diff:1.1E}')
        #np.testing.assert_array_almost_equal(G_iw.data, G_iw_fit.data, decimal=tol*0.1)

        Sigma_iw_fit = G_iw_fit.copy()
        Sigma_iw_fit << inverse(G0_iw) - inverse(G_iw_fit)
        Sigma_iw_diff = np.max(np.abs(Sigma_iw.data - Sigma_iw_fit.data))
        print(f'Sigma_iw_diff = {Sigma_iw_diff:1.1E}')
        #np.testing.assert_array_almost_equal(Sigma_iw.data, Sigma_iw_fit.data, decimal=tol*10)

        if verbose:
            G_laa = d.tau_from_dlr(G_xaa)
            tau_l = d.get_tau(beta)

            from triqs.plot.mpl_interface import oplot, oplotr, oploti, plt

            plt.figure(figsize=(12, 12))
            subp = [4, 2, 1]

            plt.subplot(*subp); subp[-1] += 1
            oplotr(G0_tau)
            oploti(G0_tau, '--')
            plt.ylabel(r'$G_0(\tau)$')

            plt.subplot(*subp); subp[-1] += 1
            oplotr(-G_tau, alpha=0.25)
            oploti(-G_tau, '--', alpha=0.25)
            oplotr(-G_tau_fit)
            oploti(-G_tau_fit, '--')
            for i, j in itertools.product(range(2), repeat=2):
                plt.plot(tau_l, -G_laa[:, i, j].real, '+')
                plt.plot(tau_l, -G_laa[:, i, j].imag, 'x')
            plt.ylabel(r'$G(\tau)$')
            plt.semilogy([], [])

            plt.legend(loc='best')

            plt.subplot(*subp); subp[-1] += 1
            oplot(G0_iw, label='G0')
            plt.subplot(*subp); subp[-1] += 1
            oplot(G_iw, label='G')

            plt.subplot(*subp); subp[-1] += 1
            oplotr(Sigma_iw_fit, '.')
            oplotr(Sigma_iw)
            plt.ylabel(r'Re[$\Sigma$]')

            plt.subplot(*subp); subp[-1] += 1
            oploti(Sigma_iw_fit, '.')
            oploti(Sigma_iw)
            plt.ylabel(r'Im[$\Sigma$]')

            plt.subplot(*subp); subp[-1] += 1
            oplotr(Sigma_iw - Sigma_iw_fit, '.')
            plt.ylabel(r'Re[$\Delta \Sigma$]')

            plt.subplot(*subp); subp[-1] += 1
            oploti(Sigma_iw - Sigma_iw_fit, '.')
            plt.ylabel(r'Im[$\Delta \Sigma$]')

            plt.tight_layout()
            plt.show()
            
    
if __name__ == '__main__':
    test_fit(verbose=False)
