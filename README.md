# slam

### Environment
 - Python 3.7

### Virtual Environment
 - Create virtual environment
 ```sh
 $ virtualenv -p python venv
 ```
 - Activate virtual environment
 ```sh
 $ source ./venv/bin/activate
 ```
 - Install all dependencies
 ```sh
 $ pip install -r requirements.txt
 ```
 - Exit virtual environment
 ```sh
 $ deactivate
 ```

### Create Kernel for run_test.ipynb
```sh
$ pip install ipykernel
$ python -m ipykernel install --user --name=gesture
```

### Place Test Data
- place test data to "test_data" folder

### Run Test Data
 - Open run_test.ipynb with jupter notebook, and run all cells.  
 ```sh
 $ jupyter notebook
 ```