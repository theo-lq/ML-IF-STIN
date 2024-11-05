#Lauch with bokeh serve ... .py

import numpy as np
import bokeh.plotting


from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Slider, RadioButtonGroup
from bokeh.layouts import column, row



n_points = 500
start = -3
stop = 3

f = lambda x: x ** 2
df = lambda x: 2 * x



##Gradient descent
def gradient_descent(start, iterations, gradient, learning_rate, momentum=0):

    def make_step(value, velocity, momentum=momentum):
        velocity = momentum * velocity + (1 - momentum) * gradient(value)
        value = value - learning_rate * velocity
        return value, velocity

    steps = [start]
    value = start
    value, velocity = make_step(value, 0, momentum=0)
    steps.append(value)

    for iteration in range(iterations-1):
        value, velocity = make_step(value, velocity)
        steps.append(value)

    return steps



def get_method(label):
    if label == 0 or label == 1:
        return gradient_descent
    else:
        return gradient_descent


def get_function(label):
    if label == 0:
        f = lambda x: x ** 2
        df = lambda x: 2 * x
        return f, df
    else:
        f = lambda x: 0.25 * ((x+1)**3) * (0.5*x-1)
        df = lambda x: 0.25 * ((x+1)**2) * (3*(0.5*x-1) + 0.5*(x+1))
        return f, df



##Widgets
function_selector = RadioButtonGroup(labels=["Convexe", "Non convexe"], active=0)
iterations_slider = Slider(title="Iterations", value=5, start=1, end=100, step=1)
start_point_slider = Slider(title="Point de départ", value=-2, start=-3, end=3, step=0.1)
learning_rate_slider = Slider(title="Learning rate", value=0.1, start=0, end=1.05, step=0.01)
momentum_slider = Slider(title="Momentum", value=0.9, start=0, end=1, step=0.05)
method_selector = RadioButtonGroup(labels=["Gradient Descent (GD)", "GD with Momentum"], active=0)

init_params = {"start": -2, "iterations": 10, "gradient": df, "learning_rate": 0.1}
steps = gradient_descent(**init_params)
data_gradient = dict(x=steps, y=[f(value) for value in steps])
source_gradient = ColumnDataSource(data=data_gradient)

x = np.linspace(start=start, stop=stop, num=n_points)
data_function = dict(x=x, y=f(x))
source_function = ColumnDataSource(data=data_function)



##Update data

def update_data(attrname, old, new):
    f, df = get_function(function_selector.active)
    iterations = iterations_slider.value
    method = get_method(method_selector.active)
    start_point = start_point_slider.value
    learning_rate = learning_rate_slider.value
    momentum = momentum_slider.value

    parameters = {
        "start": start_point,
        "iterations": iterations,
        "gradient": df,
        "learning_rate": learning_rate,
        "momentum": momentum
    }

    if method_selector.active == 0: parameters["momentum"] = 0

    steps = method(**parameters)
    source_gradient.data = dict(x=steps, y=[f(value) for value in steps])
    source_function.data = dict(x=x, y=f(x))




##Plot

plot = bokeh.plotting.figure()
plot.line('x', 'y', source=source_function, line_width=3, line_alpha=0.6, legend_label="Fonction", line_color='blue', line_cap='round')
plot.line('x', 'y', source=source_gradient, line_width=3, legend_label="Descente", line_color='red', line_join="round")
plot.circle('x', 'y', source=source_gradient, line_color='red', fill_color='red', size=8)



##Page

for widget in [function_selector, iterations_slider, start_point_slider, learning_rate_slider, momentum_slider, method_selector]:
    if widget == method_selector or widget == function_selector:
        widget.on_change('active', update_data)
    else:
        widget.on_change('value', update_data)


inputs = column(function_selector, iterations_slider, start_point_slider, learning_rate_slider, momentum_slider, method_selector)
curdoc().add_root(row(inputs, plot, width=1500))
curdoc().title = "Gradient Descent Visualizer"
