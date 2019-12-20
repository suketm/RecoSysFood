
# Food Recommendation System for Fast Food Chain

Our client is a fast food chain, and has data of five thousand customers. They want to increase sale to existing customers. In this project, we built a collaborative filtering model to enhance sale through cross sell of differnt items present in the menu.


## File structure
**Configuration file:** /config/config.yaml

**Python virtual environment setting file:** environment.yml

**Python files**
1. Classes for building recommendation system:
    ```
    /src/utils/customer_acquisition_functions.py
    ```
2. Script to built recommendation:
    ```
    /src/customer_acquisition.py
    ```

## Flow to execute code for modelling:
1. Create a virtual environment and activate the environment
    ```
    conda env create -f environment.yml
    activate RecoSysFood
    ```
2. Make reommendation of food items and save the output file in **'/reports/recommended_foods.pkl'**
    ```
    python customer_acquisition.py ../config/config.yaml
    ```