# File: rigidbody.py

import abc
import torch

from mechamodlearn import nn, utils
from mechamodlearn.models import CholeskyMMNet, SymmetricMMNet, PotentialNet, GeneralizedForceNet, \
    GradPotentialNet, SharedMMVEmbed


class AbstractRigidBody:

    @property
    @abc.abstractmethod
    def thetamask(self):
        """Returns theta mask for configuration q.
        These should use utils.diffangles to compute differences
        """

    @abc.abstractmethod
    def mass_matrix(self, q):
        """Return mass matrix for configuration q"""

    @abc.abstractmethod
    def potential(self, q):
        """Return potential for configuration q"""
        # NOTE: this actually models -V(q) instead of V(q), so G(q) can be obtained by taking the gradient.

    @abc.abstractmethod
    def generalized_force(self, q, v, u):
        """Return generalized force for configuration q, velocity v, external torque u"""

    def kinetic_energy(self, q, v):
        mass_matrix = self.mass_matrix(q)
        # TODO(jkg): Check if this works correctly for batched
        kenergy = 0.5 * (v.unsqueeze(1) @ (mass_matrix @ v.unsqueeze(2))).squeeze(2)
        return kenergy

    def lagrangian(self, q, v):
        """ Returns the Lagrangian of a mechanical system
        """
        kenergy = self.kinetic_energy(q, v)
        pot = self.potential(q)
        lag = kenergy - pot
        return lag

    def hamiltonian(self, q, v):
        """ Returns the Hamiltonian of a mechanical system
        """
        kenergy = self.kinetic_energy(q, v)
        pot = self.potential(q)
        ham = kenergy + pot
        return ham

    def corriolisforce(self, q, v, mass_matrix=None):
        """ Computes the corriolis matrix times v
        """
        with torch.enable_grad():
            if mass_matrix is None:
                mass_matrix = self.mass_matrix(q)

            Mv = mass_matrix @ v.unsqueeze(2)

            KE = 0.5 * v.unsqueeze(1) @ Mv

            Cv_KE = torch.autograd.grad(KE.sum(), q, retain_graph=True, create_graph=True)[0]

            gMv = torch.stack([
                torch.autograd.grad(Mv[:, i].sum(), q, retain_graph=True, create_graph=True)[0]
                for i in range(q.size(1))
            ], dim=1)

            Cv = gMv @ v.unsqueeze(2) - Cv_KE.unsqueeze(2)

        return Cv

    def corriolis(self, q, v, mass_matrix=None):
        """ Computes the corriolis matrix
        """
        with torch.enable_grad():
            if mass_matrix is None:
                mass_matrix = self.mass_matrix(q)

            qdim = q.size(1)
            B = mass_matrix.size(0)

            mass_matrix = mass_matrix.reshape(-1, qdim, qdim)

            # TODO vectorize
            rows = []

            for i in range(qdim):
                cols = []
                for j in range(qdim):
                    qgrad = torch.autograd.grad(
                        torch.sum(mass_matrix[:, i, j]), q, retain_graph=True, create_graph=True)[0]
                    cols.append(qgrad)

                rows.append(torch.stack(cols, dim=1))

            dMijk = torch.stack(rows, dim=1)

        corriolis = 0.5 * ((dMijk + dMijk.transpose(2, 3) - dMijk.transpose(1, 3)
                           ) @ v.reshape(B, 1, qdim, 1)).squeeze(3)
        return corriolis

    def gradpotential(self, q):
        """ Returns the conservative forces acting on the system
        """
        with torch.enable_grad():
            pot = self.potential(q)
            gvec = torch.autograd.grad(torch.sum(pot), q, retain_graph=True, create_graph=True)[0]
        return gvec

    def solve_euler_lagrange(self, q, v, u=None):
        """ Computes `qddot` (generalized acceleration) by solving
        the Euler-Lagrange equation (Eq 7 in the paper)
        \qddot = M^-1 (F - Cv - G)
        """
        with torch.enable_grad():
            with utils.temp_require_grad((q, v)):
                M = self.mass_matrix(q)
                Cv = self.corriolisforce(q, v, M)
                G = self.gradpotential(q)

        F = torch.zeros_like(Cv)

        if u is not None:
            F = self.generalized_force(q, v, u)

        # Solve M \qddot = F - Cv - G
        qddot = torch.solve(F - Cv - G.unsqueeze(2), M)[0].squeeze(2)
        return qddot
    
    def solve_euler_lagrange_from_F(self, q, v, F):
        """Computes `qddot`, assuming the force is directly controlled."""
        with torch.enable_grad():
            with utils.temp_require_grad((q, v)):
                M = self.mass_matrix(q)
                Cv = self.corriolisforce(q, v, M)
                G = self.gradpotential(q)
                
        # Solve M \qddot = F - Cv - G
        qddot = torch.solve(F.unsqueeze(2) - Cv - G.unsqueeze(2), M)[0].squeeze(2)
        return qddot
    
    def inverse_dynamics(self, q, v, qddotd, M=None, Cv=None, G=None):
        """ Computes generalized force using
        the Euler-Lagrange equation
        F = M\qddot + Cv + G
        """
        with torch.enable_grad():
            with utils.temp_require_grad((q, v)):
                if M is None:
                    M = self.mass_matrix(q)
                if Cv is None:
                    Cv = self.corriolisforce(q, v, M)
                if G is None:
                    G = self.gradpotential(q)

        F = (M @ qddotd.unsqueeze(2) + Cv + G.unsqueeze(2)).squeeze(2)
        return F


class LearnedRigidBody(AbstractRigidBody, torch.nn.Module):

    def __init__(self, qdim: int, udim: int, thetamask: torch.tensor, mass_matrix=None,
                 potential=None, generalized_force=None, hidden_sizes=None):
        """

        Arguments:
        - `qdim`:
        - `udim`: [int]
        - `thetamask`: [torch.Tensor (1, qdim)] 1 if angle, 0 otherwise
        - `mass_matrix`: [torch.nn.Module]
        - `potential`: [torch.nn.Module]
        - `generalized_force`: [torch.nn.Module]
        - hidden_sizes: [list]
        """
        self._qdim = qdim
        self._udim = udim

        self._thetamask = thetamask

        super().__init__()

        if mass_matrix is None:
            mass_matrix = CholeskyMMNet(qdim, hidden_sizes=hidden_sizes)

        self._mass_matrix = mass_matrix

        if potential is None:
            potential = PotentialNet(qdim, hidden_sizes=hidden_sizes)

        self._potential = potential

        if generalized_force is None:
            generalized_force = GeneralizedForceNet(qdim, udim, hidden_sizes)

        self._generalized_force = generalized_force

    def mass_matrix(self, q):
        return self._mass_matrix(q)

    def potential(self, q):
        return self._potential(q)

    def generalized_force(self, q, v, u):
        return self._generalized_force(q, v, u)

    @property
    def thetamask(self):
        return self._thetamask

    def forward(self, q, v, u=None):
        return self.solve_euler_lagrange(q, v, u)

    
class DeLaN(AbstractRigidBody, torch.nn.Module):

    def __init__(self, qdim: int, thetamask: torch.tensor, mass_matrix_type=CholeskyMMNet,
                 hidden_sizes=None):
        """

        Arguments:
        - `qdim`: [int]
        - `thetamask`: [torch.Tensor (1, qdim)] 1 if angle, 0 otherwise
        - `mass_matrix_type`: [class]
        - hidden_sizes: [list]
        """
        self._qdim = qdim
        self._udim = qdim

        self._thetamask = thetamask

        super().__init__()
        
        embed = SharedMMVEmbed(qdim, hidden_sizes=hidden_sizes)
        self._embed = embed

        mass_matrix = mass_matrix_type(qdim, embed=self._embed)

        self._mass_matrix = mass_matrix

        gradpotential = GradPotentialNet(qdim, embed=self._embed)

        self._gradpotential = gradpotential
        
        # Assume generalized force is direct input
        self._generalized_force = lambda q, v, u: u

    def mass_matrix(self, q):
        return self._mass_matrix(q)

    def gradpotential(self, q):
        return self._gradpotential(q)
    
    def generalized_force(self, q, v, u):
        return self._generalized_force(q, v, u)

    @property
    def thetamask(self):
        return self._thetamask

    def forward(self, q, v, qddotd):
        return self.inverse_dynamics(q, v, qddotd)
    
    def solve_euler_lagrange(self, q, v, u=None):
        """Should apply F directly by using solve_euler_lagrange from F."""
        raise NotImplementedError