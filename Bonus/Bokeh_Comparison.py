#Lauch with bokeh serve ... .py


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Slider, RadioButtonGroup
from bokeh.layouts import column, row


n_points = 500

f = lambda x: np.cos(x) - np.tanh(x)

x = np.linspace(0, 8, n_points)
y = f(x) + 0.2*np.random.normal(size=(n_points))
X = pd.DataFrame({"x": x})  


def learn_predict(model, parameters):
    machine = model(**parameters).fit(X, y)
    y_predicted = machine.predict(X)
    return y_predicted
    
    
    
    



source = ColumnDataSource(data=dict(x=x, y_predicted=learn_predict(DecisionTreeRegressor, {"max_depth": 4})))


plot = figure()
plot.line(x, f(x), line_width=3, line_alpha=0.6, legend_label="Original", line_color='blue')
plot.line('x', 'y_predicted', source=source, line_width=3, legend_label="Predicted", line_color='red')




model_label = RadioButtonGroup(labels=["Decision Tree", "Random Forest", "XGBoost"], active=0)
depth = Slider(title="Max_depth", value=4, start=1, end=10, step=1)
estimators = Slider(title="Number of estimators", value=10, start=1, end=50, step=1)
learning_rate_slider = Slider(title="Learning rate", value=0.3, start=0.01, end=1, step=0.01)







def get_model(label):
    if label == 0: return DecisionTreeRegressor
    elif label == 1: return RandomForestRegressor
    else: return XGBRegressor

    


    
def update_data(attrname, old, new):
    max_depth = depth.value
    model = get_model(model_label.active)
    n_estimators = estimators.value
    learning_rate = learning_rate_slider.value

    parameters = {
        "n_estimators": n_estimators,
        "max_depth": max_depth
        }
    
    if model_label.active == 0: del parameters["n_estimators"]
    if model_label.active == 2: parameters["learning_rate"] = learning_rate
    
    y_predicted = learn_predict(model, parameters)

    source.data = dict(x=x, y_predicted=y_predicted)






for widget in [depth, model_label, estimators, learning_rate_slider]:
    if widget == model_label:
        widget.on_change('active', update_data)
    else:
        widget.on_change('value', update_data)
    




inputs = column(model_label, depth, estimators, learning_rate_slider)
curdoc().add_root(row(inputs, plot, width=800))
curdoc().title = "Regressor Visualizer"