import os
import numpy as np
import datetime
from geostatspy import geostats
from deap import creator, base, algorithms, tools
from random import uniform, randint, choice
from scoop import futures
import warnings

warnings.filterwarnings("ignore")
start = datetime.datetime.now()
##############################################
# INPUT PARAMETERS
##############################################
suffix =
population_size = 100
number_of_generations = 10
lag_dist_model = 8.5

crossover_probability = 0.9
mutation_probability = 0.1
children_produced = int(population_size * .75)
indiv_selected = int(children_produced * .54)
nug_min, nug_max = 0.0, 0.7  # min and max nugget. Change it only if the optimization does not converge

xlag_ = 0.05
half_distance =  # increase 50% the semivariogram

##############################################
# Do not modify the code below
##############################################
nlags = int(half_distance / xlag_)
azi_rang_tensor = np.load(os.path.join(os.getcwd(), 'results', "azi_rang_tensor" + suffix + ".npy"))
semiv_tensor = np.load(os.path.join(os.getcwd(), 'results', "semivar_tensor" + suffix + ".npy"))


# @jit(nopython=True, parallel=True)
def _cova2(x1, y1, x2, y2, nst, cc, aa, it, anis, rotmat, maxcov):
    """
    Simplified version from geostatspy.

    Args:
        x1:
        y1:
        x2:
        y2:
        nst:
        cc:
        aa:
        it:
        anis:
        rotmat:
        maxcov:

    Returns:
    """
    EPSLON = 0.000000
    # Check for very small distance
    dx = x2 - x1
    dy = y2 - y1
    # the sum of two squared numbers is never zero
    if (dx * dx + dy * dy) < EPSLON:
        cova2_ = maxcov
        return cova2_

    # Non-zero distance, loop over all the structures
    cova2_ = 0.0
    for js in range(0, nst):
        # Compute the appropriate structural distance
        dx1 = dx * rotmat[0, js] + dy * rotmat[1, js]
        dy1 = (dx * rotmat[2, js] + dy * rotmat[3, js]) / anis[js]
        h = np.sqrt(max((dx1 * dx1 + dy1 * dy1), 0.0))
        if it[js] == 1:
            # Spherical model
            hr = h / aa[js]
            if hr < 1.0:
                cova2_ += cc[js] * (1.0 - hr * (1.5 - 0.5 * hr * hr))
        elif it[js] == 2:
            # Exponential model
            cova2_ += cc[js] * np.exp(-3.0 * h / aa[js])
        elif it[js] == 3:
            # Gaussian model
            hh = -3.0 * (h * h) / (aa[js] * aa[js])
            cova2_ += cc[js] * np.exp(hh)

    return cova2_


# @jit(nopython=True, parallel=True)
def _vmodel(nlag, xlag, azm, vario):
    """
    GSLIB's VMODEL program (Deutsch and Journel, 1998) converted from the
    original Fortran to Python by Michael Pyrcz, the University of Texas at
    Austin (Mar, 2019).
    """

    # Parameters
    DEG2RAD = np.pi / 180.0

    # Declare arrays
    index = np.zeros(nlag + 1)
    h = np.zeros(nlag + 1)
    gam = np.zeros(nlag + 1)
    cov = np.zeros(nlag + 1)
    ro = np.zeros(nlag + 1)

    # Load the variogram
    nst = vario.get("nst")
    cc = np.zeros(nst)
    aa = np.zeros(nst)
    it = np.zeros(nst)
    ang = np.zeros(nst)
    anis = np.zeros(nst)

    c0 = vario.get("nug")
    cc[0] = vario.get("cc1")
    it[0] = vario.get("it1")
    ang[0] = vario.get("azi1")
    aa[0] = vario.get("hmaj1")
    anis[0] = vario.get("hmin1") / vario.get("hmaj1")
    if nst == 2:
        cc[1] = vario.get("cc2")
        it[1] = vario.get("it2")
        ang[1] = vario.get("azi2")
        aa[1] = vario.get("hmaj2")
        anis[1] = vario.get("hmin2") / vario.get("hmaj2")

    xoff = np.sin(DEG2RAD * azm) * xlag
    yoff = np.cos(DEG2RAD * azm) * xlag
    rotmat, maxcov = geostats.setup_rotmat(c0, nst, it, cc, ang, 99999.9)

    xx = 0.0
    yy = 0.0
    for il in range(0, nlag + 1):
        index[il] = il
        cov[il] = _cova2(0.0, 0.0, xx, yy, nst, cc, aa, it, anis, rotmat, maxcov)
        gam[il] = maxcov - cov[il]
        ro[il] = cov[il] / maxcov
        h[il] = np.sqrt(max((xx * xx + yy * yy), 0.0))
        xx = xx + xoff
        yy = yy + yoff

    return h, gam


# @jit(nopython=True)
def _make_variogram(nug, nst, it1, cc1, azi1, hmaj1, hmin1, it2=1, cc2=0, azi2=0, hmaj2=0, hmin2=0):
    if cc2 == 0:
        nst = 1
    var = dict(
        [
            ("nug", nug),
            ("nst", nst),
            ("it1", it1),
            ("cc1", cc1),
            ("azi1", azi1),
            ("hmaj1", hmaj1),
            ("hmin1", hmin1),
            ("it2", it2),
            ("cc2", cc2),
            ("azi2", azi2),
            ("hmaj2", hmaj2),
            ("hmin2", hmin2),
        ]
    )
    return var


# @jit(nopython=True)
def _model_semivariog(individual, azi_maj, azi_min, range_maj, range_min):
    semivariog = {"nug": individual[5], "nst": 2, "it1": individual[0], "cc1": individual[1], "azi1": azi_maj,
                  "hmaj1": individual[2], "hmin1": individual[3], "it2": individual[4],
                  "cc2": abs(1.00 - individual[5] - individual[1]), "azi2": azi_min, "hmaj2": range_maj,
                  "hmin2": range_min}

    return semivariog


def largest_lag(_maj_tensor):
    largest_dist = np.max(_maj_tensor[:, 0])
    return largest_dist


# @jit(nopython=True, parallel=True)
def _theoretical_semivariog(individual):
    semivariog_model = _model_semivariog(individual, maj_azi, min_azi, maj_range, min_range)
    semivariog = _make_variogram(**semivariog_model)

    _, gamma_maj = _vmodel(nlags, xlag_, azm=semivariog_model.get("azi1"), vario=semivariog)
    _, gamma_min = _vmodel(nlags, xlag_, azm=semivariog_model.get("azi2"), vario=semivariog)

    return gamma_maj, gamma_min


def _mutation(individual):
    gene = randint(0, len(individual) - 2)  # select which parameter to mutate
    # structure model
    if gene in [0, 4]:
        if individual[gene] == 1:  # if spherical
            individual[gene] = randint(2, 3)
        elif individual[gene] == 2:  # if exponential
            individual[gene] = choice([1, 3])
        else:  # if Gaussian
            individual[gene] = randint(1, 2)

    elif gene == 1:
        individual[1] = uniform(cc1_min, cc1_max)

    elif gene == 2:
        individual[2] = uniform(hmaj1_min, hmaj1_max)

    elif gene == 3:
        individual[3] = uniform(hmin1_min, hmin1_max)

    elif gene == 5:
        individual[5] = uniform(nug_min, nug_max)

    return individual,


##############################################
# Set up
##############################################
maj_azi = azi_rang_tensor[0, 0]
maj_range = azi_rang_tensor[0, 1]
maj_tensor = semiv_tensor[0, ...]

# mask to remove zero gamma values
mask = maj_tensor[1:, 0] != 0
mask_final = np.zeros((len(mask) + 1))
mask_final[0] = True
mask_final[1:] = mask
maj_tensor = maj_tensor[mask_final.astype('bool')]


# min_azi = azi_rang_tensor[1, 0]
min_azi = azi_rang_tensor[0, 0]
min_range = azi_rang_tensor[1, 1]
min_tensor = semiv_tensor[1, ...]
min_tensor = min_tensor[mask_final.astype('bool')]

# variogram modeling
# nug_min, nug_max = 0.0, 1.0
cc1_min, cc1_max = 0.0, 1.0
hmaj1_min, hmaj1_max = 0, maj_range
hmin1_min, hmin1_max = 0, min_range
max_n_lags = largest_lag(maj_tensor)

N_CYCLES = 1

##############################################
# Genetic algorithm setup
##############################################
# use a negative sign to specify minimization
creator.create("SemivarioMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.SemivarioMin)

toolbox = base.Toolbox()

# define_indv_and_pop
# gene 0: structure 1 model
toolbox.register("it1", randint, 1, 3)
# gene 1: variance contribution for structure 1
toolbox.register("cc1", uniform, cc1_min, cc1_max)
# gene 2: range for major semivariogram for structure 1
toolbox.register("hmaj1", uniform, hmaj1_min, hmaj1_max)
# gene 3: range for minor semivariogram for structure 1
toolbox.register("hmin1", uniform, hmin1_min, hmin1_max)
# gene 4: structure 2 model
toolbox.register("it2", randint, 1, 3)
# gene 5: nugget effect
toolbox.register("nugget", uniform, nug_min, nug_max)

toolbox.register(
    "individual", tools.initCycle, creator.Individual,
    (toolbox.it1, toolbox.cc1, toolbox.hmaj1, toolbox.hmin1, toolbox.it2, toolbox.nugget,),
    n=N_CYCLES
)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# loss function
def _loss_function(gamma_array, gamma_experimental):
    indices = np.round(gamma_experimental[:, 0] / lag_dist_model)
    gamma_theoretical = gamma_array[indices.astype(int)]
    gamma_experimentl = gamma_experimental[:, 1]

    w_hi = gamma_experimental[:, 2]
    sd_e = np.std(gamma_experimentl)
    N = gamma_experimental.shape[0]

    squares = np.square(gamma_experimentl - gamma_theoretical)
    suma = np.sum(np.square(w_hi) * squares)
    loss = np.sqrt(suma / N) / sd_e

    return loss


def _compute_loss(individual):
    gamma_maj, gamma_min = _theoretical_semivariog(individual)
    # loss
    major_dir_loss = _loss_function(gamma_maj, maj_tensor)
    minor_dir_loss = _loss_function(gamma_min, min_tensor)
    total_fit_loss = 0.5 * (major_dir_loss + minor_dir_loss)

    # contributions must add up to one
    vario_model = _model_semivariog(individual, maj_azi, min_azi, maj_range, min_range)
    cc1 = vario_model.get("cc1")
    cc2 = vario_model.get("cc2")
    nugget = vario_model.get("nug")
    var_loss = abs(1. - cc1 - cc2 - nugget)

    return var_loss, total_fit_loss,


# operators
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", _mutation)
toolbox.register("evaluate", _compute_loss)
toolbox.register("select", tools.selNSGA2)

if __name__ == '__main__':
    toolbox.register("map", futures.map)

    population = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    # stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("max", np.max, axis=0)

    algorithms.eaMuPlusLambda(
        population=population,
        toolbox=toolbox,
        mu=indiv_selected,
        lambda_=children_produced,
        cxpb=crossover_probability,
        mutpb=mutation_probability,
        stats=stats,
        ngen=number_of_generations,
        halloffame=hof,
        verbose=True
    )

    semivario_model = _model_semivariog(hof[0], maj_azi, min_azi, maj_range, min_range)
    finish = datetime.datetime.now()
    delta_time = finish - start
    print(''.join(['{0}: {1}; '.format(k, np.round(v, 2)) for k, v in semivario_model.items()]))
    print(f"Seconds: {delta_time.total_seconds():.2f}")
