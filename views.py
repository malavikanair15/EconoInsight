# data_analysis_app/views.py
import os
from django.shortcuts import render, HttpResponse
from data_analysis_project import settings
import pandas as pd
from ydata_profiling import ProfileReport
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
import base64
from io import StringIO
import pickle

def data_tab(request):
    # Read Titanic dataset from CSV
    df = pd.read_csv("/Users/maheshk/Downloads/Data analytics project")
 # Get the first ten rows of the dataset
    df_first_ten = df.head(10)
    df_last_ten = df.tail(10)

    # Get information about the columns (data types, non-null counts, null counts)
    columns_info = pd.DataFrame({
        'Column Name': df.columns,
        'Data Type': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Null Count': df.isnull().sum()
    })

    # Convert DataFrame to HTML for rendering in template
    table_html1 = df_first_ten.to_html(classes='table table-hover table-bordered')
    table_html2 = df_last_ten.to_html(classes='table table-hover table-bordered')

    # Include the columns information in the HTML template
    columns_info_html = f"{columns_info.to_html(classes='table table-hover table-bordered', index=False)}"

    return render(request, 'data_tab.html', {'table_html1': table_html1, 'table_html2':table_html2, 'columns_info_html': columns_info_html})



def profile_view(request):
    df = pd.read_csv("C/Users/maheshk/Downloads/Data analytics project data.csv")

    # Create a profile report
    profile = ProfileReport(df, title="Pandas Profiling Report")
    
    print('Setting Dir Path')
    print(settings.TEMPLATE_DIR)
    templates_dir = settings.TEMPLATE_DIR
    report_path = os.path.join(templates_dir, 'report.html')

    # Save the report to the templates directory
    profile.to_file(report_path)
    # Pass the HTML file path to the template

    return render(request, 'report.html', {'report_path': report_path})


  
def descriptive_statistics_tab(request):
    # Read Titanic dataset from CSV
    df = pd.read_csv("/Users/maheshk/Downloads/Data analytics project data.csv")
    
    # Perform descriptive statistics using pandas
    descriptive_stats = df.describe().to_html(classes='table table-bordered table-hover')

    return render(request, 'descriptive_statistics_tab.html', {'descriptive_stats': descriptive_stats})




def box_plot(request):
    # Read Titanic dataset from CSV
    df = pd.read_csv("/Users/maheshk/Downloads/Data analytics project data.csv")

    # Default settings for the plot
    default_category = 'Pclass'
    default_value = 'Fare'

    # Get user-selected options (if any)
    selected_category = request.GET.get('category', default_category)
    selected_value = request.GET.get('value', default_value)

    # Validate selected features
    if selected_category not in df.columns or selected_value not in df.columns:
        error_message = "Invalid features selected for box plot."
        return render(request, 'error_page.html', {'error_message': error_message})

    # Create an interactive box plot using Plotly
    fig = px.box(df, x=selected_category, y=selected_value, title=f'Box Plot: {selected_value} by {selected_category}')

    # Convert the plot to HTML for rendering in the template
    plot_html = fig.to_html(full_html=False)

    # Pass parameters to the template for customization options
    box_cus_options = {
        'categories': df.columns.tolist(),
        'default_category': default_category,
        'default_value': default_value,
        'selected_category': selected_category,
        'selected_value': selected_value,
    }
    return {'plot_html': plot_html, 'box_cus_options': box_cus_options}
    #return render(request, 'box_plot.html', {'plot_html': plot_html, 'customization_options': customization_options})










import pandas as pd
import plotly.express as px
from django.shortcuts import render

def exploratory_data_analysis_tab(request):
    # Read the new dataset from CSV
    df = pd.read_csv("/Users/maheshk/Downloads/Data analytics project data.csv")

    # Default settings for the plot
    default_feature = 'GDP'
    default_bins = 20

    # Get user-selected options (if any)
    selected_feature = request.GET.get('feature', default_feature)
    selected_bins = int(request.GET.get('bins', default_bins))

    # Create an interactive histogram using Plotly
    fig = px.histogram(df, x=selected_feature, nbins=selected_bins, title=f'{selected_feature} Distribution')

    # Convert the plot to HTML for rendering in the template
    plot_html = fig.to_html(full_html=False)

    # Pass parameters to template for customization options
    customization_options = {
        'features': df.columns.tolist(),
        'default_feature': default_feature,
        'default_bins': default_bins,
        'selected_feature': selected_feature,
        'selected_bins': selected_bins,
    }

    # Default settings for the map
    default_category = 'Country'
    default_value = 'GDP'

    # Get user-selected options (if any)
    selected_category = request.GET.get('map_cat', default_category)
    selected_value = request.GET.get('map_val', default_value)

    # Validate selected features
    if selected_category not in df.columns or selected_value not in df.columns:
        error_message = "Invalid features selected for the map."
        return render(request, 'error_page.html', {'error_message': error_message})

    # Create an interactive map using Plotly
    fig_map = px.scatter_geo(
        df,
        locations='Country',  # Assuming 'Country' is the column containing country names
        locationmode='country names',
        color=selected_value,
        hover_name='Country',
        title=f'Map: {selected_value} by {selected_category}',
        color_continuous_scale='Viridis',
    )

    # Convert the plot to HTML for rendering in the template
    map_plot_html = fig_map.to_html(full_html=False)

    # Pass parameters to the template for customization options
    map_cus_options = {
        'categories': df['Country'].tolist(),
        'default_category': default_category,
        'default_value': default_value,
        'selected_category': selected_category,
        'selected_value': selected_value,
    }

    # Scatter Plot
    # Default settings for the scatter plot
    default_x_feature = 'GDP'
    default_y_feature = 'Inflation'

    # Get user-selected options (if any)
    selected_x_feature = request.GET.get('x_feature', default_x_feature)
    selected_y_feature = request.GET.get('y_feature', default_y_feature)

    # Create an interactive scatter plot using Plotly
    fig_scatter = px.scatter(df, x=selected_x_feature, y=selected_y_feature, color='Country',
                             title=f'Scatter Plot: {selected_x_feature} vs. {selected_y_feature}')

    # Convert the scatter plot to HTML for rendering in the template
    plot_html_scatter = fig_scatter.to_html(full_html=False)

    # Pass parameters to template for customization options
    customization_options_scatter = {
        'features': df.columns.tolist(),
        'default_x_feature': default_x_feature,
        'default_y_feature': default_y_feature,
        'selected_x_feature': selected_x_feature,
        'selected_y_feature': selected_y_feature,
    }
    import statsmodels.api as sm
   # Regression Analysis
    # Select independent variables
    independent_vars = ['Inflation', 'FDI', 'Exchange rate', 'Population', 'Money Supply']

    # Add a constant term for the intercept
    X = sm.add_constant(df[independent_vars])

    # Dependent variable
    y = df['GDP']

    # Fit the regression model
    model = sm.OLS(y, X).fit()

    # Get the regression summary
    regression_summary = model.summary()

    # Convert the summary to HTML for rendering in the template
    regression_html = regression_summary.tables[1].as_html()

    # Pass regression results to the template
    regression_results = {
        'regression_html': regression_html,
    }

    return render(request, 'exploratory_data_analysis_tab.html', {
    'plot_html': plot_html,
    'customization_options': customization_options,
    'map_plot_html': map_plot_html,
    'map_cus_options': map_cus_options,
    'plot_html_scatter': plot_html_scatter,
    'customization_options_scatter': customization_options_scatter,
    'regression_results': regression_results,
})


import pickle

def predict_gdp(request):
    if request.method == 'POST':
        model_filename = 'gdp_prediction_model.pkl'
        model_path = os.path.join(os.path.dirname(__file__), 'models', model_filename)
        
        # Load the trained model from the pickle file
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)

        # Get user input from the form
        Inflation = float(request.POST.get('inflation'))
        FDI = float(request.POST.get('FDI'))
        exchange_rate = float(request.POST.get('exchange_rate'))
        population = float(request.POST.get('population'))
        money_supply = float(request.POST.get('money_supply'))

        # Make prediction
        input_data = [[Inflation, FDI, exchange_rate, population, money_supply]]
        prediction = model.predict(input_data)[0]

        # Render the result
        return render(request, 'prediction_result.html', {'prediction': prediction})

    # If the form is not submitted, render the empty form
    return render(request, 'predict_gdp.html', {'prediction': None})


