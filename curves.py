import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import ttkbootstrap as ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
def curveFinder(df: pd.DataFrame):
    canvasesCurve = []

    # Define curve functions
    def curve_function(type, x, *params):
        if type == "Linear":
            a, b = params
            return a * x + b
        elif type == "Quadrática":
            a, b, c = params
            return a * x**2 + b * x + c
        elif type == "Logarítmica":
            a, b = params
            return a * np.log(x) + b
        elif type == "Senoide":
            a, b, c = params
            return a * np.sin(b * x + c)
        elif type == "Cosenoide":
            a, b, c = params
            return a * np.cos(b * x + c)
        elif type == "Tangente":
            a, b = params
            return a * np.tan(b * x)
        else:
            a, b = params
            return a * x + b

    def on_curve_type_change(selected_type):
        for canvas in canvasesCurve:
            canvas.destroy()

        # Update the curve type and fit the data
        x = df[selected_x.get()].values
        y = df.iloc[:, -1].values  # Assume Y-axis data is the last column

        # Define the curve fitting function
        def fit_func(x, *params):
            return curve_function(selected_type, x, *params)

        # Initial guess for parameters
        param_count = {
            "Linear": 2,
            "Quadrática": 3,
            "Logarítmica": 2,
            "Senoide": 3,
            "Cosenoide": 3,
            "Tangente": 2,
        }
        initial_guess = [1] * param_count[selected_type]

        try:
            # Fit the data
            params, _ = curve_fit(fit_func, x, y, p0=initial_guess)
        except Exception as e:
            print(f"Error in curve fitting: {e}")
            return

        # Generate predictions
        x_fit = np.linspace(x.min(), x.max(), 500)
        y_pred = fit_func(x_fit, *params)

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_facecolor("#002B36")
        fig.set_facecolor("#002B36")
        ax.spines["bottom"].set_color("#FFFFFF")
        ax.spines["top"].set_color("#FFFFFF")
        ax.spines["left"].set_color("#FFFFFF")
        ax.spines["right"].set_color("#FFFFFF")
        ax.tick_params(axis='x', colors='#FFFFFF')
        ax.tick_params(axis='y', colors='#FFFFFF')
        ax.yaxis.label.set_color('#FFFFFF')
        ax.xaxis.label.set_color('#FFFFFF')
        ax.title.set_color("#FFFFFF")

        # Plot the data
        ax.scatter(x, y, c='yellow', label='Real Data')
        ax.plot(x_fit, y_pred, label='Fitted Curve', color='red', lw=2)
        ax.set_xlabel(selected_x.get())
        ax.set_ylabel(df.columns[-1])
        ax.set_title(f"{selected_x.get()} vs {df.columns[-1]}")
        ax.legend()

        # Integrate plot into Tkinter
        canvas = FigureCanvasTkAgg(fig, master=rootCurve)
        canvas_widget = canvas.get_tk_widget()
        canvasesCurve.append(canvas_widget)
        canvas_widget.grid(row=1, column=0, rowspan=18, columnspan=18, sticky='nsew')
        canvas.draw()

    def on_x_change(*args):
        # Update the plot on x-axis selection change
        on_curve_type_change(selected_curve.get())

    # Tkinter window setup
    rootCurve = ttk.Toplevel()
    rootCurve.title("LabAI - Curvas")

    # Variables for selection
    selected_x = ttk.StringVar(value=df.columns[0])
    selected_curve = ttk.StringVar(value="Linear")

    # Dropdowns for axis and curve type selection
    xType = ttk.OptionMenu(rootCurve, selected_x, *[""] + list(df.columns[:-1]), command=on_x_change)
    xType.grid(row=0, column=0)

    curveType = ttk.OptionMenu(
        rootCurve, selected_curve, *["", "Linear", "Quadrática", "Logarítmica", "Senoide", "Cosenoide", "Tangente"], command=on_curve_type_change
    )
    curveType.grid(row=0, column=1)

    # Initialize plot
    on_curve_type_change("Linear")

    rootCurve.mainloop()