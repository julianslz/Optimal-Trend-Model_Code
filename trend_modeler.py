import os
import datetime
import numpy as np
import pandas as pd
from geostatspy.GSLIB import DataFrame2ndarray
from astropy.convolution import Gaussian2DKernel, convolve
from deap import creator, base, algorithms, tools
from random import uniform, randint
from scoop import futures
import warnings

warnings.filterwarnings("ignore")
start = datetime.datetime.now()
##############################################
# INPUT PARAMETERS
##############################################
xcoor =
ycoor =
feature =
target_mean = 0.962
target_var = 1.020

cell_size = 1
min_window = 5
max_window = 30
min_theta = 0
max_theta = 180

# genetic algorithm hyperparameters
starting_population = 30
number_of_generations = 15
crossover_probability = 0.9
mutation_probability = 0.10
children_produced = int(starting_population * 0.6)
indiv_selected = int(children_produced * 0.6)

# dataset
df = pd.read_csv(".csv")


##############################################
# Do not modify the code below
##############################################

def save_results(_hof):
    tm_results = np.zeros((len(_hof), 5))
    for i in range(len(_hof)):
        tm_results[i, 0] = _hof[i][0]
        tm_results[i, 1] = _hof[i][1]
        tm_results[i, 2] = _hof[i][2]
        tm_results[i, 3] = _hof[i].fitness.values[0]
        tm_results[i, 4] = _hof[i].fitness.values[1]

    hover_text = []
    for rowi in range(len(tm_results)):
        hover_text.append((
                "Mean loss: {ml:.2f}<br>" +
                "Variance loss: {vl:.2f}<br>" +
                "X size: {xs:.2f}<br>" +
                "Y size: {ys:.2f}<br>" +
                "Theta:  {th:.2f}<br>"
        ).format(
            ml=tm_results[rowi, 3],
            vl=tm_results[rowi, 4],
            xs=tm_results[rowi, 0],
            ys=tm_results[rowi, 1],
            th=tm_results[rowi, 2],
        ))

    tm_results = pd.DataFrame(
        tm_results, columns=['X size', 'Y size', 'Theta', 'Mean loss', 'Variance loss']
    )
    tm_results['Text'] = hover_text

    return tm_results


def pareto_eq(ind1, ind2):
    return np.all(ind1.fitness.values == ind2.fitness.values)


def _convolve_2d(x_window, y_window, theta):
    """
    Convolve the sparse data with a Gaussian kernel
    :param x_window: Window size in the X direction
    :type x_window: float
    :param y_window: Window size in the Y direction
    :type y_window: float
    :return:
    """
    theta *= (3.1416 / 180)
    kernel = Gaussian2DKernel(x_window, y_window, theta)
    trend_array = convolve(
        df_grid,
        kernel,
        boundary='extend',
        nan_treatment='interpolate',
        normalize_kernel=True
    )

    return trend_array


def _mapping_to_cells(array, dataset):
    """
    Get the trend values at each cell using their cell ID.
    Parameters
    ----------
    array
    dataset

    Returns
    -------

    """
    trend = np.flipud(array).ravel()
    cells = dataset['Cell id'].to_numpy()
    trend_vector = trend[cells]

    return trend_vector


def _max_and_min(dataset, xcor, ycor, _cell_size):
    """
    Compute the maximum and minimum values of a rectangle for modeling
    """
    # get the range in each dimension
    range_x = np.round(np.ptp(dataset[xcor]))
    range_y = np.round(np.ptp(dataset[ycor]))
    half = np.abs(range_x - range_y) / 2
    # add additional length to the smaller axis so the cell size is a multiple of the total length
    if range_x > range_y:
        ymin_ = np.round(dataset[ycor].min()) - half
        ymax_ = np.round(dataset[ycor].max()) + half
        xmax_ = np.round(dataset[xcor].max())
        xmin_ = np.round(dataset[xcor].min())
    else:
        xmin_ = np.round(dataset[xcor].min()) - half
        xmax_ = np.round(dataset[xcor].max()) + half
        ymax_ = np.round(dataset[ycor].max())
        ymin_ = np.round(dataset[ycor].min())

    if (xmax_ - xmin_) % _cell_size > 0:  # the cell size is not exact
        cells = int(np.ceil((xmax_ - xmin_) / _cell_size))
        half = (cells * _cell_size - (xmax_ - xmin_)) / 2
        ymin_ -= half
        ymax_ += half
        xmax_ += half
        xmin_ -= half

    return xmin_, xmax_, ymin_, ymax_


def data_cleaner(dataset, xcor, ycor, _cell_size, min_x, max_y, min_y):
    xi = (np.floor((dataset[xcor] - min_x) / _cell_size)).astype(int)
    yi = (np.floor((dataset[ycor] - min_y) / _cell_size)).astype(int)
    ncells = int((max_y - min_y) / _cell_size)
    dataset['Cell id'] = ncells * yi + xi

    return dataset


def mutater(individual):
    gene = randint(0, len(individual) - 2)  # select which parameter to mutate
    if gene in [0, 1]:  # window sizes
        individual[gene] = uniform(min_window, max_window)
    else:
        individual[2] = uniform(min_theta, max_theta)

    return individual,


##############################################
# Set up
##############################################
df.dropna(inplace=True)
df = df[[xcoor, ycoor, feature]]
N_CYCLES = 1

# get the variance and mean of the feature of interest
xmin, xmax, ymin, ymax = _max_and_min(df, xcoor, ycoor, cell_size)
# xmin = 424000
# xmax = 427239.4
# ymin = 5.404275e6
# ymax = 5.410388e6
df = data_cleaner(df, xcoor, ycoor, cell_size, xmin, ymax, ymin)

df_grid = DataFrame2ndarray(df, xcoor, ycoor, feature, xmin, xmax, ymin, ymax, cell_size)

##############################################
# Genetic algorithm setup
##############################################
# use a negative sign to specify minimization
creator.create("TrendFitness", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.TrendFitness)
toolbox = base.Toolbox()

# define_indv_and_pop
# gene 0
toolbox.register("xwindow", uniform, min_window, max_window)
# gene 1
toolbox.register("ywindow", uniform, min_window, max_window)
# gene 2
toolbox.register("theta", uniform, min_theta, max_theta)

# register the individual
toolbox.register(
    "individual", tools.initCycle, creator.Individual, (toolbox.xwindow, toolbox.ywindow, toolbox.theta),
    n=N_CYCLES
)
# define the population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def trend_operations(x_window, y_window, theta, dataset, target):
    # do convolution to get the trend model
    trend_array = _convolve_2d(x_window, y_window, theta)
    trend_vector = _mapping_to_cells(trend_array, dataset)  # trend only at variance locations

    trend_mean = np.nanmean(trend_array)

    residuals_vector = dataset[target] - trend_vector
    covariance_trend_residuals = (np.cov(trend_vector, residuals_vector)[1, 0])
    variance_observed = np.var(trend_vector) + np.var(residuals_vector) + 2 * covariance_trend_residuals

    return trend_mean, variance_observed


def loss_function(individual):
    mean_observed, var_observed = trend_operations(
        x_window=individual[0], y_window=individual[1], theta=individual[2], dataset=df, target=feature
    )
    mean_loss = (np.abs(target_mean - mean_observed) / target_mean) * 100
    var_loss = (np.abs(target_var - var_observed) / target_var) * 100

    return mean_loss, var_loss,


# operators
toolbox.register("evaluate", loss_function)
# toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=min_window, up=max_window, eta=20.0)
# toolbox.register("mutate", tools.mutPolynomialBounded, low=min_window, up=max_window, eta=20.0, indpb=0.03)
toolbox.register("mate", tools.cxTwoPoint)
# toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.03)
toolbox.register("mutate", mutater)
toolbox.register("select", tools.selNSGA2)

# ################
if __name__ == '__main__':
    toolbox.register("map", futures.map)

    population = toolbox.population(n=starting_population)
    # hof = tools.HallOfFame(1)
    hof = tools.ParetoFront()
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

    finish = datetime.datetime.now()
    delta_time = finish - start
    print(f"Running time: {delta_time.total_seconds():.0f} seconds.")
    print("Saving results...")
    results = save_results(hof)
    results.to_pickle(os.path.join(os.getcwd(), "results", "tm_results.pkl"))
