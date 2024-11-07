# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot
import numpy
import scipy
from scipy.optimize import minimize
import scipy.io


def real_to_complex(z):
    return z[0] + 1j * z[1]


def complex_to_real(z):
    return numpy.array([numpy.real(z), numpy.imag(z)])


class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.f(*args)
        # .. todo:: deepcopy here if returning objects
        return self.memo[args]


def compute_alpha(omega, material_name):
    """
    .. warning: $w = 2 \pi f$
    w is called circular frequency
    f is called frequency
    :param material:
    """
    
    material_properties = materials[material_name]
    # Use the material properties passed in instead of hardcoded values
    phi = material_properties["phi"]
    gamma_p = material_properties["gamma_p"]
    sigma = material_properties["sigma"]
    rho_0 = material_properties["rho_0"]
    alpha_h = material_properties["alpha_h"]
    c_0 = material_properties["c_0"]

    # Remove the hardcoded Birch LT values
    # phi = 0.529  # porosity
    # gamma_p = 7.0 / 5.0
    # sigma = 151429.0  # resitivity
    # rho_0 = 1.2
    # alpha_h = 1.37  # tortuosity
    # c_0 = 340.0

    # parameters of the geometry
    L = 0.01

    # parameters of the mesh
    resolution = 12  # := number of elements along L

    # parameters of the material (cont.)
    mu_0 = 1.0
    ksi_0 = 1.0 / (c_0**2)
    mu_1 = phi / alpha_h
    ksi_1 = phi * gamma_p / (c_0**2)
    a = sigma * (phi**2) * gamma_p / ((c_0**2) * rho_0 * alpha_h)

    ksi_volume = phi * gamma_p / (c_0**2)
    a_volume = sigma * (phi**2) * gamma_p / ((c_0**2) * rho_0 * alpha_h)
    mu_volume = phi / alpha_h
    k2_volume = (
        (1.0 / mu_volume)
        * ((omega**2) / (c_0**2))
        * (ksi_volume + 1j * a_volume / omega)
    )
    print(k2_volume)

    # parameters of the objective function
    A = 1.0
    B = 1.0

    # defining k, omega and alpha dependant parameters' functions
    @Memoize
    def lambda_0(k, omega):
        if k**2 >= (omega**2) * ksi_0 / mu_0:
            return numpy.sqrt(k**2 - (omega**2) * ksi_0 / mu_0)
        else:
            return numpy.sqrt((omega**2) * ksi_0 / mu_0 - k**2) * 1j

    @Memoize
    def lambda_1(k, omega):
        temp1 = (omega**2) * ksi_1 / mu_1
        temp2 = numpy.sqrt((k**2 - temp1) ** 2 + (a * omega / mu_1) ** 2)
        real = (1.0 / numpy.sqrt(2.0)) * numpy.sqrt(k**2 - temp1 + temp2)
        im = (-1.0 / numpy.sqrt(2.0)) * numpy.sqrt(temp1 - k**2 + temp2)
        return complex(real, im)

    @Memoize
    def g(y):
        # .. warning:: not validated ***********************
        return 1.0

    @Memoize
    def g_k(k):
        # .. warning:: not validated ***********************
        if k == 0:
            return 1.0
        else:
            return 0.0

    @Memoize
    def f(x, k):
        return (lambda_0(k, omega) * mu_0 - x) * numpy.exp(-lambda_0(k, omega) * L) + (
            lambda_0(k, omega) * mu_0 + x
        ) * numpy.exp(lambda_0(k, omega) * L)

    @Memoize
    def chi(k, alpha, omega):
        return g_k(k) * (
            (lambda_0(k, omega) * mu_0 - lambda_1(k, omega) * mu_1)
            / f(lambda_1(k, omega) * mu_1, k)
            - (lambda_0(k, omega) * mu_0 - alpha) / f(alpha, k)
        )

    @Memoize
    def eta(k, alpha, omega):
        return g_k(k) * (
            (lambda_0(k, omega) * mu_0 + lambda_1(k, omega) * mu_1)
            / f(lambda_1(k, omega) * mu_1, k)
            - (lambda_0(k, omega) * mu_0 + alpha) / f(alpha, k)
        )

    @Memoize
    def e_k(k, alpha, omega):
        expm = numpy.exp(-2.0 * lambda_0(k, omega) * L)
        expp = numpy.exp(+2.0 * lambda_0(k, omega) * L)

        if k**2 >= (omega**2) * ksi_0 / mu_0:
            return (
                (A + B * (numpy.abs(k) ** 2))
                * (
                    (1.0 / (2.0 * lambda_0(k, omega)))
                    * (
                        (numpy.abs(chi(k, alpha, omega)) ** 2) * (1.0 - expm)
                        + (numpy.abs(eta(k, alpha, omega)) ** 2) * (expp - 1.0)
                    )
                    + 2
                    * L
                    * numpy.real(
                        chi(k, alpha, omega) * numpy.conj(eta(k, alpha, omega))
                    )
                )
                + B
                * numpy.abs(lambda_0(k, omega))
                / 2.0
                * (
                    (numpy.abs(chi(k, alpha, omega)) ** 2) * (1.0 - expm)
                    + (numpy.abs(eta(k, alpha, omega)) ** 2) * (expp - 1.0)
                )
                - 2
                * B
                * (lambda_0(k, omega) ** 2)
                * L
                * numpy.real(chi(k, alpha, omega) * numpy.conj(eta(k, alpha, omega)))
            )
        else:
            return (
                (
                    (A + B * (numpy.abs(k) ** 2))
                    * (
                        L
                        * (
                            (numpy.abs(chi(k, alpha, omega)) ** 2)
                            + (numpy.abs(eta(k, alpha, omega)) ** 2)
                        )
                        + complex(0.0, 1.0)
                        * (1.0 / lambda_0(k, omega))
                        * numpy.imag(
                            chi(k, alpha, omega)
                            * numpy.conj(eta(k, alpha, omega) * (1.0 - expm))
                        )
                    )
                )
                + B
                * L
                * (numpy.abs(lambda_0(k, omega)) ** 2)
                * (
                    (numpy.abs(chi(k, alpha, omega)) ** 2)
                    + (numpy.abs(eta(k, alpha, omega)) ** 2)
                )
                + complex(0.0, 1.0)
                * B
                * lambda_0(k, omega)
                * numpy.imag(
                    chi(k, alpha, omega)
                    * numpy.conj(eta(k, alpha, omega) * (1.0 - expm))
                )
            )

    @Memoize
    def sum_e_k(omega):
        def sum_func(alpha):
            s = 0.0
            for n in range(-resolution, resolution + 1):
                k = n * numpy.pi / L
                s += e_k(k, alpha, omega)
            return s

        return sum_func

    @Memoize
    def alpha(omega):
        alpha_0 = numpy.array(complex(40.0, -40.0))
        temp = real_to_complex(
            minimize(
                lambda z: numpy.real(sum_e_k(omega)(real_to_complex(z))),
                complex_to_real(alpha_0),
                tol=1e-4,
            ).x
        )
        print(temp, "------", "je suis temp")
        return temp

    @Memoize
    def error(alpha, omega):
        temp = numpy.real(sum_e_k(omega)(alpha))
        return temp

    temp_alpha = alpha(omega)
    temp_error = error(temp_alpha, omega)

    return temp_alpha, temp_error


def run_compute_alpha(material_properties):
    print("Computing alpha...")
    numb_omega = 10  # 1000
    omegas = numpy.linspace(2.0 * numpy.pi, numpy.pi * 10000, num=numb_omega)
    temp = [
        compute_alpha(omega, material_properties=material_properties)
        for omega in omegas
    ]
    print("temp:", "------", temp)
    alphas, errors = map(list, zip(*temp))
    alphas = numpy.array(alphas)
    errors = numpy.array(errors)

    print("Writing alpha...")
    output_filename = "dta_omega_" + str(material) + ".mtx"
    scipy.io.mmwrite(
        output_filename,
        omegas.reshape(alphas.shape[0], 1),
        field="complex",
        symmetry="general",
    )
    output_filename = "dta_alpha_" + str(material) + ".mtx"
    scipy.io.mmwrite(
        output_filename,
        alphas.reshape(alphas.shape[0], 1),
        field="complex",
        symmetry="general",
    )
    output_filename = "dta_error_" + str(material) + ".mtx"
    scipy.io.mmwrite(
        output_filename,
        errors.reshape(errors.shape[0], 1),
        field="complex",
        symmetry="general",
    )

    return


def run_plot_alpha(material):
    color = "darkblue"

    print("Reading alpha...")
    input_filename = "dta_omega_" + str(material) + ".mtx"
    omegas = scipy.io.mmread(input_filename)
    omegas = omegas.reshape(omegas.shape[0])
    input_filename = "dta_alpha_" + str(material) + ".mtx"
    alphas = scipy.io.mmread(input_filename)
    alphas = alphas.reshape(alphas.shape[0])
    input_filename = "dta_error_" + str(material) + ".mtx"
    errors = scipy.io.mmread(input_filename)
    errors = errors.reshape(errors.shape[0])

    print("Plotting alpha...")
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.subplot(1, 1, 1)
    matplotlib.pyplot.plot(numpy.real(omegas), numpy.real(alphas), color=color)
    matplotlib.pyplot.xlabel(r"$\omega$")
    matplotlib.pyplot.ylabel(r"$\operatorname{Re}(\alpha)$")
    matplotlib.pyplot.ylim(0, 35)
    # matplotlib.pyplot.show()
    matplotlib.pyplot.savefig("fig_alpha_real_" + str(material) + ".jpg")
    matplotlib.pyplot.close(fig)

    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.subplot(1, 1, 1)
    matplotlib.pyplot.plot(numpy.real(omegas), numpy.imag(alphas), color=color)
    matplotlib.pyplot.xlabel(r"$\omega$")
    matplotlib.pyplot.ylabel(r"$\operatorname{Im}(\alpha)$")
    matplotlib.pyplot.ylim(-120, 10)
    # matplotlib.pyplot.show()
    matplotlib.pyplot.savefig("fig_alpha_imag_" + str(material) + ".jpg")
    matplotlib.pyplot.close(fig)

    fig = matplotlib.pyplot.figure()
    ax = matplotlib.pyplot.axes()
    ax.fill_between(numpy.real(omegas), numpy.real(errors), color=color)
    matplotlib.pyplot.ylim(1.0e-9, 1.0e-4)
    matplotlib.pyplot.yscale("log")
    matplotlib.pyplot.xlabel(r"$\omega$")
    matplotlib.pyplot.ylabel(r"$e(\alpha)$")
    # matplotlib.pyplot.show()
    matplotlib.pyplot.savefig("fig_error_" + str(material) + ".jpg")
    matplotlib.pyplot.close(fig)

    return


def run():
    # Define materials and their properties
    materials = {
        "Wood": {
            "phi": 0.5,
            "gamma_p": 7.0 / 5.0,  # Keeping this constant for all materials
            "sigma": 12500.0,
            "rho_0": 600.0,
            "alpha_h": 1.35,
            "c_0": 360.0,
        },
        "Polyester": {
            "phi": 0.9,
            "gamma_p": 7.0 / 5.0,
            "sigma": 20000.0,
            "rho_0": 40.0,
            "alpha_h": 1.2,
            "c_0": 340.0,
        },
        "Melamine": {
            "phi": 0.95,
            "gamma_p": 7.0 / 5.0,
            "sigma": 13000.0,
            "rho_0": 10.0,
            "alpha_h": 1.3,
            "c_0": 340.0,
        },
        "Wool": {
            "phi": 0.9,
            "gamma_p": 7.0 / 5.0,
            "sigma": 22500.0,
            "rho_0": 100.0,
            "alpha_h": 1.4,
            "c_0": 340.0,
        },
    }

    # Calculate alpha for each material
    colors = ["blue", "red", "green", "purple"]
    numb_frequence = 1000  # Increased for smoother curves
    frequences = numpy.linspace(2.0, 2000, num=numb_frequence)
    omegas = 2 * numpy.pi * frequences

    # Create figures for real and imaginary parts
    fig_real = matplotlib.pyplot.figure(figsize=(10, 6))
    fig_imag = matplotlib.pyplot.figure(figsize=(10, 6))

    for material_name, properties in materials.items():
        # Fix: Pass properties instead of material_name and use material_properties parameter
        temp = [
            compute_alpha(omega, material_properties=properties) for omega in omegas
        ]
        alphas, _ = map(list, zip(*temp))
        alphas = numpy.array(alphas)

        # Plot real part
        matplotlib.pyplot.figure(fig_real.number)
        matplotlib.pyplot.plot(
            numpy.real(frequences), numpy.real(alphas), label=material_name
        )

        # Plot imaginary part
        matplotlib.pyplot.figure(fig_imag.number)
        matplotlib.pyplot.plot(
            numpy.real(frequences), numpy.imag(alphas), label=material_name
        )

    # Finalize real part plot
    matplotlib.pyplot.figure(fig_real.number)
    matplotlib.pyplot.xlabel(r"frequence")
    matplotlib.pyplot.ylabel(r"$\operatorname{Re}(\alpha)$")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.savefig("alpha_real_comparison.jpg")

    # Finalize imaginary part plot
    matplotlib.pyplot.figure(fig_imag.number)
    matplotlib.pyplot.xlabel(r"frequence")
    matplotlib.pyplot.ylabel(r"$\operatorname{Im}(\alpha)$")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.savefig("alpha_imag_comparison.jpg")

    matplotlib.pyplot.close("all")
    return


if __name__ == "__main__":
    run()
    print("End.")


materials = {
        "Wood": {
            "phi": 0.5,
            "gamma_p": 7.0 / 5.0,  # Keeping this constant for all materials
            "sigma": 12500.0,
            "rho_0": 600.0,
            "alpha_h": 1.35,
            "c_0": 360.0,
        },
        "Polyester": {
            "phi": 0.9,
            "gamma_p": 7.0 / 5.0,
            "sigma": 20000.0,
            "rho_0": 40.0,
            "alpha_h": 1.2,
            "c_0": 340.0,
        },
        "Melamine": {
            "phi": 0.95,
            "gamma_p": 7.0 / 5.0,
            "sigma": 13000.0,
            "rho_0": 10.0,
            "alpha_h": 1.3,
            "c_0": 340.0,
        },
        "Wool": {
            "phi": 0.9,
            "gamma_p": 7.0 / 5.0,
            "sigma": 22500.0,
            "rho_0": 100.0,
            "alpha_h": 1.4,
            "c_0": 340.0,
        },
    }
