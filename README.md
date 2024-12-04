# data_viz_for_engineers_hf

This is the Home Work of Balázs Márk Mihályi for the class *Data Visualization for Engineers*. The project includes an interactive Dash app showcasing the results and a profiling analysis of the data processing steps.

## Installation

Follow the steps below to set up the project on your local machine:

1. **Clone the repository**  
   Clone this repository to your local machine using the following command:  
   ```bash
   git clone https://github.com/your-username/data_viz_for_engineers_hf.git
   ```
2. **Create virtual environment**
   Navigate into the project directory and create a virtual environment:
   ```bash
   cd data_viz_for_engineers_hf
   python -m venv venv
   ```
3. **Activate virtual environment**
- On Windows:
   ```bash
   venv\Scripts\activate
   ```
- On Linux:
    ```bash
    source venv/bin/activate
    ```
4. Install dependencies
   Install the required packages using ```requirements.txt```
   ```bash
   pip install -r requirements.txt
   ```

## Running the Dash App

To run the interactive Dash app that displays the results:
1. **Ensure the virtual environment is active**
   Activate the virtual environment as described in the installation steps.
2. **Run the app**
   Use the following command to start the Dash app:
   ```bash
   python app.py
   ```
3. **Access the app**
   Open your web browser and navigate to http://127.0.0.1:8050 to view the application.

## Profiling

This project includes profiling analysis to optimize and evaluate the performance of the data processing pipeline. The profiling results are available in two formats:
- Jupyter Notebook: Open and review profiling.ipynb for a detailed interactive profiling analysis. You can run it in any Jupyter Notebook environment.
- HTML Report: View profiling.html in your web browser for a static version of the profiling results.

To generate the profiling report yourself, ensure you have the required packages installed and follow the steps in profiling.ipynb.
