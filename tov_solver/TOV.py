import warnings
from typing import TYPE_CHECKING, Dict, Optional, Callable, Self
from datetime import datetime

import numpy as np
import h5py as h5
from scipy.integrate import solve_ivp, simps, cumulative_trapezoid
from scipy.integrate._ivp.ivp import OdeResult
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, toms748

from tabulatedEOS.PizzaEOS import RU, U, PizzaEOS

if TYPE_CHECKING:
    from numpy.typing import NDArray

fourpi = 4 * np.pi
_lin_interp_keys = ['enthalpy', 'mu_nu', 'ye',
                    'Gamma', 'cs2', 'energy', ]


def calculate_tidal_love_number(y, c):
    """
    Eq. (23) in Hinderer etal (2008)
    (note that there is an Erratum for this equation)
    """
    return (
        8/5 * c**5 * (1 - 2 * c) ** 2 * (2 + 2 * c * (y - 1) - y)
        / (
            2 * c * (6 - 3 * y + 3 * c * (5 * y - 8))
            + 4 * c**3
            * (13 - 11 * y + c * (3 * y - 2) + 2 * c**2 * (1 + y))
            + 3 * (1 - 2 * c) ** 2
            * (2 - y + 2 * c * (y - 1)) * np.log(1 - 2 * c)
        )
    )


class TOVSolution:
    """
    Attributes:
        parameters (dict): A dictionary of the TOV paramters obtained from solving.
        data (dict): A dictionary of the radial arrays obtained from solving.
        eos_table (dict): A dictionary containing the cold beta-eq. EOS table.
    """
    data: Dict[str, "NDArray[np.float_]"]
    parameters: Dict[str, float]
    eos_table: Dict[str, "NDArray[np.float_]"]
    eos_namee: str

    @classmethod
    def load(cls, path: str) -> Self:
        instance = cls.__new__(cls)
        with h5.File(path, 'r') as hf:
            instance.data = {}
            for key in hf.keys():
                if key == 'eos_table':
                    continue
                instance.data[key] = np.array(hf[key])
            instance.parameters = {}
            for key in hf.attrs.keys():
                if isinstance(par := hf.attrs[key], float):
                    instance.parameters[key] = par
            instance.eos_table = {}
            for key in hf['eos_table'].keys():
                instance.eos_table[key] = np.array(hf['eos_table'][key])
            instance.eos_name = hf['eos_table'].attrs['eos_name']
        return instance

    def __init__(
        self,
        solution: OdeResult,
        solver: "TOVSolver",
        central_enthalpy: float,
        by_pressure: bool = False,  # DEPRECATED
    ):
        """
        """
        self.eos_table = solver.eos_table
        self.eos_name = solver.eos_name
        self.parameters = {}
        self.data = {}

        self.data["r"], self.data["m"], self.data["y"] = solution.y

        if by_pressure:
            # DEPRECATED
            # in this case the "central_enthalpy" is actually the central pressure
            self.parameters["central_pressure"] = central_enthalpy
            self.parameters["central_enthalpy"] = \
                solver._interpolate_eos_table(
                central_enthalpy, 'enthalpy', x_key="press")

            self.data["r"] = self.data["r"] ** 0.5
            self.data["press"] = solution.t

            for key in "rho eps cs2 energy ye enthalpy".split():
                self.data[key] = np.array([
                    solver._interpolate_eos_table(pp, key, x_key="press")
                    for pp in self.data["press"]
                ])
        else:
            self.parameters["central_enthalpy"] = central_enthalpy
            self.parameters["central_pressure"] = \
                solver._interpolate_eos_table(
                central_enthalpy, 'press', x_key="enthalpy")

            self.data["enthalpy"] = solution.t

            for key in "rho eps cs2 energy ye press".split():
                self.data[key] = np.array(
                    [solver._interpolate_eos_table(hh, key, x_key="enthalpy")
                     for hh in self.data["enthalpy"]]
                )

        self.parameters["R"] = self.data["r"][-1]
        self.parameters["M"] = self.data["m"][-1]

        c = self.parameters["M"] / self.parameters["R"]
        k_2 = calculate_tidal_love_number(y=self.data["y"][-1], c=c)

        self.parameters['Mbary'] = self._calculate_mbary()
        self.parameters["C"] = c
        self.parameters["k_2"] = k_2
        self.parameters["Lambda"] = 2/3*k_2*c**-5

    def save(self, path: str,):
        with h5.File(path, 'w') as hf:
            for key in self.data:
                hf.create_dataset(key, data=self.data[key])
            for key in self.parameters:
                hf.attrs[key] = self.parameters[key]
                hf.attrs['EOS'] = self.eos_name
            hf.attrs['date'] = datetime.today().strftime('%Y-%m-%d')
            grp = hf.create_group('eos_table')
            grp.attrs['eos_name'] = self.eos_name
            for key in self.eos_table:
                grp.create_dataset(key, data=self.eos_table[key])

    def _calculate_mbary(self) -> float:
        rad = self.data['r']
        rho = self.data['rho']
        mm = self.data['m']
        lam = - np.log(1 - 2*mm/rad)

        integ = 4*np.pi*rad**2*rho*np.exp(lam/2)

        return float(simps(integ, rad))


class TOVSolver:
    """
    A class for solving the Tolman-Oppenheimer-Volkoff (TOV) equations for a given equation of state (EOS).

    Parameters:
        equation_of_state (PizzaEOS): The equation of state in Pizza format.
        verbose (bool, optional): If True, print progress and information during solving. Default is False.
        correct_pressure (bool, optional): If True, correct any non-monotonic pressure in the EOS table. Default is True.

    Attributes:
        eos_table (dict): The one-dimensional eos table used for solving.
    """

    eos_table: Dict[str, "NDArray[np.float_]"]
    eos_name: str
    pizza_table: Dict[str, "NDArray[np.float_]"]
    solutions_by_mass: Dict[float, TOVSolution]
    solutions_by_enthalpy: Dict[float, TOVSolution]
    Mmax_enthalpy: Optional[float] = None

    def __init__(
        self,
        eos: PizzaEOS,
        verbose: bool = False,
        correct_pressure: bool = True,
    ):
        """
        Initializes the TOV solver with the given equation of state.

        Args:
            equation_of_state (PizzaEOS): The equation of state in Pizza format.
            verbose (bool, optional): If True, print progress and information during solving. Default is False.
            correct_pressure (bool, optional): If True, correct any non-monotonic pressure in the EOS table. Default is True.
        """
        self.pizza_table = {}
        self.verbose = verbose
        self.eos_name = eos.name
        self._generate_eos_table(eos)
        if correct_pressure:
            self._correct_pressure()
        self._calc_enthalpy()
        self._fix_negative_values()

        self.solutions_by_mass = {}
        self.solutions_by_enthalpy = {}

    def _generate_eos_table(self, eos: PizzaEOS,):
        """
        Generate the EOS table required for TOV solving.
        """
        self.eos_table = {}
        self.eos_table["rho"] = eos.get_key("density")
        self.eos_table["ye"] = np.zeros_like(self.eos_table["rho"])
        ye_0 = eos.get_key("ye")
        ye_0 = ye_0[ye_0 <= .5]  # equilibrium ye should always be smaller .5
        max_ye = ye_0.max()
        min_ye = ye_0.min()
        cutoff = 1e-5
        rho, ye = np.meshgrid(self.eos_table["rho"], ye_0, indexing="ij")
        getter = eos.get_cold_caller(
            ["mu_p", "mu_e", "mu_n"],
            lambda mu_p, mu_e, mu_n, **_: mu_p + mu_e - mu_n
        )
        mu_nu = getter(ye=ye, rho=rho)

        for ii, (dd, mn) in enumerate(zip(self.eos_table["rho"], mu_nu)):
            if np.all(mn > 0):
                self.eos_table["ye"][ii] = min_ye
                continue
            if np.all(mn < 0):
                self.eos_table["ye"][ii] = max_ye
                continue

            f = interp1d(ye_0, mn, kind="linear", bounds_error=True)
            try:
                self.eos_table["ye"][ii] = toms748(
                    f, a=ye_0[0], b=ye_0[-1],
                    xtol=1e-6, maxiter=100
                )
            except ValueError:
                if dd > cutoff:
                    self.eos_table["ye"][ii] = min_ye
                else:
                    self.eos_table["ye"][ii] = max_ye

        kw = dict(
            ye=self.eos_table["ye"],
            rho=self.eos_table["rho"]
        )

        self.eos_table["mu_nu"] = getter(**kw)
        for table_key, eos_key in (('press', 'pressure'),
                                   ('eps', 'internalEnergy'),
                                   ('cs2', 'cs2')):
            getter = eos.get_cold_caller([eos_key])
            self.eos_table[table_key] = getter(**kw)

        self.eos_table["energy"] = self.eos_table["rho"] * \
            (1 + self.eos_table["eps"])
        self.eos_table["Gamma"] = self.eos_table['energy'] + \
            self.eos_table['press']
        self.eos_table["Gamma"] *= self.eos_table['cs2'] / \
            self.eos_table['press']

    def _correct_pressure(self):
        """
        Correct any non-monotonic pressure in the EOS table.
        """
        unphys_mask = self.eos_table["press"] > 20
        if self.verbose:
            print(
                f"Correcting {unphys_mask.sum()} unphysical pressure values."
            )
        if np.any(unphys_mask):
            i_start = np.where(unphys_mask)[0]
            for kk in self.eos_table:
                self.eos_table[kk] = np.delete(
                    self.eos_table[kk], i_start
                )

        monotonic_mask = np.diff(self.eos_table["press"]) <= 0
        if self.verbose:
            print(
                f"Correcting {monotonic_mask.sum(
                )} non-monotonic pressure values."
            )
        if np.any(monotonic_mask):
            i_start = np.where(monotonic_mask)[0] + 1
            for kk in self.eos_table:
                self.eos_table[kk] = np.delete(
                    self.eos_table[kk], i_start
                )

    def _calc_enthalpy(self):
        pp = self.eos_table["press"]
        integ = 1/(self.eos_table["energy"] + pp)
        pp_l = np.logspace(np.log10(pp[0]), np.log10(pp[-1]), 5000)
        integ_l = _logint(pp, integ, pp_l)
        h_l = cumulative_trapezoid(integ_l, pp_l, initial=integ_l[0]*pp_l[0])

        self.eos_table['enthalpy'] = _logint(pp_l, h_l, pp)

    def _fix_negative_values(self):
        for key in "energy enthalpy".split():
            if not np.all(mask := (self.eos_table[key] >= 0)):
                if self.verbose:
                    warnings.warn(
                        f"{self}: {(~mask).sum()} points in {
                            key} are either <=0 or NaN",
                        RuntimeWarning
                    )
                self.eos_table[key][~mask] = self.eos_table[key][mask].min()

        for key in "cs2 ".split():
            if not np.all(mask := (self.eos_table[key] > 0)):
                if self.verbose:
                    warnings.warn(
                        f"{self}: {(~mask).sum()} points in {
                            key} are either <=0 or NaN",
                        RuntimeWarning
                    )
                self.eos_table[key][~mask] = self.eos_table[key][mask].min()

    def _read_pizza_table(self, eos):
        """
        Get the pizza table for EOS.
        """
        self.pizza_table = {}
        pizza_path = eos.hydro_path.replace("hydro.h5", "pizza")
        (
            self.pizza_table["rho"],
            self.pizza_table["eps"],
            self.pizza_table["press"],
        ) = np.loadtxt(pizza_path, unpack=True, skiprows=5, usecols=(0, 1, 2))
        self.pizza_table["rho"] *= RU.rho / 1e3
        self.pizza_table["press"] *= RU.press * 10

        i_table = self.eos_table["rho"].argmin()
        min_rho = self.eos_table["rho"][i_table]
        i_pizza = np.argmin(np.abs(self.pizza_table["rho"] - min_rho))
        eps_offset = (self.pizza_table["eps"]
                      [i_pizza] - self.eos_table["eps"][i_table])
        self.pizza_table["eps"] -= eps_offset

    def _interpolate_eos_table(
        self,
        value: float,
        key: str,
        x_key: str,
    ):
        """
        Interpolate the EOS table for the given value using the specified keys.

        Args:
            value (float): Value to interpolate.
            key (str): Key to interpolate in the EOS table.
            x_key (str): Key representing the x-coordinate (pressure or
                pseudo enthalpy) for interpolation.

        Returns:
            float: Interpolated value for the given value and key.
        """
        if value <= self.eos_table[x_key][0]:
            return self.eos_table[key][0]

        tabulated = self.eos_table[key]
        log_x = x_key not in _lin_interp_keys
        log_interp = key not in _lin_interp_keys

        x_tabulted = self.eos_table[x_key]
        if log_x:
            x_tabulted = np.log10(x_tabulted)
            value = np.log10(value)
        if log_interp:
            tabulated = np.log10(tabulated)

        res = interp1d(
            x_tabulted,
            tabulated,
            kind="linear",
            bounds_error=True
        )(value)

        if log_interp:
            return 10**res
        return res

    def _add_solution(
        self,
        solution: OdeResult,
        central_enthalpy: float,
        by_pressure: bool = False,  # DEPRECATED
    ):
        tov_solution = TOVSolution(
            solution=solution,
            solver=self,
            central_enthalpy=central_enthalpy,
            by_pressure=by_pressure,  # DEPRECATED
        )
        mass = np.round(tov_solution.parameters['M'], 4)
        self.solutions_by_mass[mass] = tov_solution
        c_enthalpy = np.round(tov_solution.parameters['central_enthalpy'], 6)
        self.solutions_by_enthalpy[c_enthalpy] = tov_solution

    def _get_sources_by_enthalpy(
        self,
        enthalpy: float,
        r_mass_y: "NDArray[np.float_]",
    ):
        """
        Calculate TOV sources as a function of pseudo enthalpy.
        See Svenja Greifs thesis for more information.

        Args:
            enthalpy (float): Pseudo enthalpy value.
            r_mass_y (Tuple[float, float, float]): Tuple containing radial coordinate, mass, and electron_fraction.

        Returns:
            np.ndarray: Array containing the derivatives of radial coordinate, mass, and electron_fraction with respect to pseudo enthalpy.
        """
        rr, mass, yy = r_mass_y

        kw = dict(value=enthalpy, x_key="enthalpy")
        energy = self._interpolate_eos_table(key="energy", **kw)
        cs2 = self._interpolate_eos_table(key="cs2", **kw)
        press = self._interpolate_eos_table(key="press", **kw)

        if rr < 0:
            return np.array([np.nan, np.nan, np.nan])
        fourpir3 = fourpi*rr**3
        rminus2m = rr - 2*mass
        denum = mass + fourpir3*press

        dr = - rr*rminus2m / denum
        dm = -fourpir3 * energy * rminus2m / denum

        dy = rminus2m * (yy + 1) * yy
        dy += (mass - fourpir3*energy) * yy
        dy += fourpir3*(5*energy + 9*press) - 6*rr
        dy += fourpir3*(energy + press) / cs2
        dy /= denum
        dy += yy
        dy -= 4*denum / rminus2m

        drmy = np.array([dr, dm, dy])

        return drmy

    def calculate_offset_values_by_enthalpy(
            self,
            central_enthalpy: float,
            dh: float):
        """
        Get the values of r, mass, and ye at a small offset dh from the central enthalpy.

        Args:
            central_enthalpy (float): Central enthalpy value.
            dh (float): Small offset value from the central enthalpy.

        Returns:
            numpy.ndarray: Array containing the values of r, mass, and ye at the offset enthalpy.
        """

        kw = dict(value=central_enthalpy, x_key="enthalpy")
        Ec = self._interpolate_eos_table(key="energy", **kw)
        cs2 = self._interpolate_eos_table(key="cs2", **kw)
        pc = self._interpolate_eos_table(key="press", **kw)

        eplusp_cs2 = (Ec + pc)/cs2
        eplus3p = Ec + 3 * pc

        r1 = (3 / (2*np.pi * eplus3p))**.5
        r3 = - r1/(4*eplus3p)
        r3 *= Ec - 3*pc - 3*eplusp_cs2 / 5
        m3 = 4*np.pi/3 * Ec * r1**3
        m5 = 4*np.pi * r1**3
        m5 *= r3*Ec/r1 - eplusp_cs2 / 5
        y2 = - 6 / (7*eplus3p)
        y2 *= Ec/3 + 11*pc + eplusp_cs2

        rdh = dh**0.5
        rdh3 = rdh**3
        rdh5 = rdh**5

        rr = r1*rdh + r3*rdh3
        mm = m3*rdh3 * + m5*rdh5
        yy = 2 + y2*dh

        return np.array([rr, mm, yy])

    def solve_by_enthalpy(
        self,
        central_enthalpy: float,
        terminal_enthalpy: Optional[float] = None,
        num_points: int = 1000,
        save: bool = True,
        **solver_kwargs,
    ) -> OdeResult:
        """
        Solve the TOV equations for a given central pseudo enthalpy.

        Args:
            central_enthalpy (float): Central pseudo enthalpy value.
            terminal_enthalpy (float, optional): Terminal pseudo enthalpy value. Default is None.
            num_points (int, optional): Number of points for evaluation. Default is 1000.
            **solver_kwargs: Additional arguments to be passed to the solver.

        Returns:
            scipy.integrate.OdeSolution: The solution of the TOV equations.
        """
        default = dict(method="DOP853", rtol=1e-6)
        solver_kwargs = {**default, **solver_kwargs}

        if terminal_enthalpy is None:
            terminal_enthalpy = self.eos_table['enthalpy'][1]

        h_span = central_enthalpy, terminal_enthalpy
        h_eval = np.linspace(
            central_enthalpy,
            terminal_enthalpy,
            num_points
        )

        y0 = self.calculate_offset_values_by_enthalpy(
            central_enthalpy, dh=central_enthalpy*1e-6)

        solution = solve_ivp(
            self._get_sources_by_enthalpy,
            y0=y0,
            t_span=h_span,
            t_eval=h_eval,
            **solver_kwargs
        )

        if solution.status < 0:
            if self.verbose:
                f"{self}: Error in solution:"
                print(solution.message)
        elif save:
            self._add_solution(solution, central_enthalpy,)
        return solution

    def _get_sources_by_pressure(self, pressure, r2_mass_ye):
        """
        Calculate TOV sources as a function of pressure.

        Args:
            pressure (float): Pressure value.
            r_squared_mass_ye (Tuple[float, float, float]): Tuple containing radial coordinate squared, mass, and electron_fraction.

        Returns:
            np.ndarray: Array containing the derivatives of radial coordinate squared, mass, and electron_fraction with respect to pressure.
        """
        r2, mass, yy = r2_mass_ye
        if any(map(lambda dd: not np.all(np.isfinite(dd)), r2_mass_ye)):
            return np.array([np.nan, np.nan, np.nan])
        if r2 < 0:
            return np.array([np.nan, np.nan, np.nan])

        kw = dict(value=pressure, x_key="press")
        energy = self._interpolate_eos_table(key="energy", **kw)
        cs2 = self._interpolate_eos_table(key="cs2", **kw)

        r3 = r2**1.5
        rr = r2**.5
        rminus2m = rr - 2*mass
        denum = (energy+pressure)*(mass + fourpi*r3*pressure)
        if denum <= 0.:
            denum = 1e-20

        dr2 = -2 * r2 * rminus2m / denum

        dm = -fourpi * r3 * energy * rminus2m / denum

        F = rr - fourpi * r3 * (energy - pressure)
        F /= rminus2m
        Q = fourpi * rr / rminus2m
        Q *= (5 * energy + 9 * pressure +
              (energy + pressure) / cs2) * r2 - 6 / fourpi
        dnudr = mass + fourpi * r3 * pressure
        dnudr /= rminus2m
        Q -= 4 * dnudr**2

        dy = -(yy**2) - yy * F - Q
        dy /= 2 * r2

        dy *= dr2

        return np.array([dr2, dm, dy])

    def solve_by_pressure(
        self,
        central_pressure: float,
        num_points: int = 1000,
        terminal_pressure: Optional[float] = None,
        save: bool = True,
        **solver_kwargs
    ) -> OdeResult:
        """
        Solve the TOV equations for a given central pressure.
        *** This method is deprecated! Use solve_by_enthalpy instead! ***

        Args:
            central_pressure (float): Central pressure value.
            num_points (int, optional): Number of points for evaluation. Default is 1000.
            terminal_pressure (float, optional): Terminal pressure value. Default is None.
            **solver_kwargs: Additional arguments to be passed to the solver.

        Returns:
            scipy.optimize.OdeSolution: The solution of the TOV equations.
        """
        warnings.warn(
            "This method will be deprecated. Use 'solve_by_enthalpy' instead.",
            DeprecationWarning
        )

        default = dict(method="DOP853", rtol=1e-6)
        solver_kwargs = {**default, **solver_kwargs}

        if terminal_pressure is None:
            terminal_pressure = 1e-17 * U.length**-2
        p_span = central_pressure, terminal_pressure

        # lin + log grid to make the integration more stable
        # p_int = central_pressure*.1
        # p_eval = np.concatenate([
        #     np.linspace(central_pressure, p_int, num_points//2+1)[:-1],
        #     np.logspace(np.log10(p_int),
        #                 np.log10(terminal_pressure),
        #                 num_points//2)
        # ])
        p_eval = np.logspace(np.log10(central_pressure),
                             np.log10(terminal_pressure),
                             100)

        p_eval = np.clip(p_eval, terminal_pressure, central_pressure)

        y0 = np.array([1e-10, 1e-10, 2.0])

        solution = solve_ivp(
            self._get_sources_by_pressure,
            y0=y0,
            t_span=p_span,
            t_eval=p_eval,
            **solver_kwargs
        )

        if solution.status < 0:
            if self.verbose:
                f"{self}: Error in solution:"
                print(solution.message)
        elif save:
            self._add_solution(solution, central_pressure, by_pressure=True)
        return solution

    def solve_for_target_mass(
        self,
        target_mass: float,
        root_finder: Callable = toms748,
        solver_kwargs: Dict = {},
        solve_by: str = "enthalpy",
        **kwargs
    ) -> float:
        """
        Find the central value for a given target mass.
        The value can either be enthalpy or pressure depending on the solve_by keyword argument.

        Args:
            target_mass (float): Target mass value.
            root_finder (callable, optional): The root-finding function to use. Default is scipy.optimize.bisect.
            solver_kwargs (dict, optional): Additional arguments to be passed to the solver function.
            solve_by (str, optional): Which solve method should be used, either "enthalpy" or "pressure" (default: "enthalpy")
            **kwargs: Additional arguments to be passed to the root-finding function.

        Returns:
            float: The central pressure corresponding to the target mass.
        """

        verbose = self.verbose

        if verbose:
            print(f"{self}: Finding central {
                  solve_by} for {target_mass}M star")
            print(f"central {solve_by:>8s} {'mass':>11s} {'radius':>11s}")
            self.verbose = False

        if solve_by == 'enthalpy':
            solver = self.solve_by_enthalpy
            a = .005
            if self.Mmax_enthalpy is not None:
                b = self.Mmax_enthalpy
            else:
                b = min(self.eos_table['enthalpy'].max(), .7)
        elif solve_by == 'pressure':
            solver = self.solve_by_pressure
            a = 5e-5
            b = 1e-3
        else:
            raise ValueError(
                f"Unknown solve_by method: {solve_by}."
                "Must be either 'enthalpy' or 'pressure'."
            )

        default = dict(a=a, b=b, rtol=1e-6, maxiter=100)
        kwargs = {**default, **kwargs}

        def _get_mass(cental_value):
            tmp_kwargs = solver_kwargs.copy()
            tmp_kwargs['save'] = False
            solution = solver(cental_value, **tmp_kwargs)
            if solution.status < 0:
                return target_mass
            if verbose:
                print(
                    f"{cental_value:16.6e} "
                    f"{solution.y[1][-1]:11.8f} "
                    f"{solution.y[0][-1]*U.length:11.8f}"
                )
            return solution.y[1][-1] - target_mass

        central_value = root_finder(_get_mass, **kwargs)

        self.verbose = verbose
        solver(central_value, **solver_kwargs)

        return central_value

    def find_maximum_mass(
        self,
        min_finder: Callable = minimize_scalar,
        solver_kwargs: Dict = {},
        solve_by: str = "enthalpy",
        **kwargs
    ) -> float:
        """
        Find the maximum TOV mass for the EOS.

        Args:
            min_finder (callable, optional): The root-finding function to use.
                Default is scipy.optimize.bisect.
            solver_kwargs (dict, optional): Additional arguments to be passed
                to the solver function.
            solve_by (str, optional): Which solve method should be used,
                either "enthalpy" or "pressure" (default: "enthalpy")
            save (bool, optional): If True, the solution will be saved in the
                solver's solutions dictionary. Default is True.
            **kwargs: Additional arguments to be passed to the
                maximum-finding function.

        Returns:
            float: The central pressure corresponding to the target mass.
        """
        verbose = self.verbose

        if verbose:
            print(
                f"{self}: Finding maximum TOV mass by solving for {solve_by}")
            print(f"central {solve_by:>8s} {'mass':>11s} {'radius':>11s}")
            self.verbose = False

        if solve_by == 'enthalpy':
            solver = self.solve_by_enthalpy
            a = .1
        elif solve_by == 'pressure':
            solver = self.solve_by_pressure
            a = 1e-4
        else:
            raise ValueError(
                f"Unknown solve_by method: {solve_by}."
                "Must be either 'enthalpy' or 'pressure'."
            )

        v_max = self.eos_table[solve_by].max()*.3
        default = dict(
            bracket=(a, v_max),
            method='bounded',
            bounds=self.eos_table['enthalpy'][[0, -1]],
        )
        if "method" in kwargs and kwargs["method"] != "bounded":
            default.pop("bounds")

        kwargs = {**default, **kwargs}

        def _get_mass(cental_value):
            tmp_kwargs = solver_kwargs.copy()
            tmp_kwargs['save'] = False
            solution = solver(cental_value, **tmp_kwargs)
            if solution.status < 0:
                return 0
            if verbose:
                print(
                    f"{cental_value:16.6e} "
                    f"{solution.y[1][-1]:11.8f} "
                    f"{solution.y[0][-1]*U.length:11.8f}"
                )
            return -solution.y[1][-1]

        result = min_finder(_get_mass, **kwargs)
        central_value = result.x

        self.verbose = verbose
        solution = solver(central_value, **solver_kwargs)
        if solution.status < 0:
            raise RuntimeError(
                f"{self}: Error in final solution for maximum Mass with {
                    solve_by} method."
                f"{solution.message}"
            )
        if not (solve_by == "pressure"):  # DEPRECATED
            self.Mmax_enthalpy = central_value
        return solution.y[1][-1]

    def __str__(self):
        return f"TOVSolver with EOS: {self.eos_name}"

    def __repr__(self):
        return self.__str__()


def _logint(x, y, xp):
    return 10**interp1d(
        np.log10(x),
        np.log10(y),
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate',
    )(
        np.log10(xp)
    )
