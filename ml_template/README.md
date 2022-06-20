# TraCon
Concept Learning to enhance Traffic Speed Prediction

This implementation stores the results inside a database. So it required to set the database by setting the necessary information are described in the following part.

Usage : 
```bash
python main.py -d defaults.ini -c config.ini
```

# Database

The templates generate two tables:
 The "config" table stores the values which are set in the configuration as well
 as an id and the time the experiment was started.
 
 The "speed results" table stores the evaluation metrics that are defined for the speed prediction, together with weights for TraCon result. 
 
# Configuration
 
 The framework is designed to handle parameters specified in config file or in the command line.
 
 Parameters can be set in three ways:
 
 1) The default.ini stores configuration entries that are the same for all experiments, e.g., db credentials
 
 2) The config.ini stores any other paramters that you want to experiment on
 
 3) In addition to the config.ini, parameters can be specified on the command line as well
 
 If a single parameter is set in multiple ways the following hierachy holds
 
 defaults < config < command line


**Given this previous point, in order to run TraCon, DB credentials should be set in default.ini, the implementation was done with PostgresDB** 

 
 # Mandatory configuration entries. 
 
 The following entries are mandatory:
 
dbHost - Url of postgres db

dbUser - Postgres db user

dbPassword - Postgres db password

dbName - Postgres db password

dbSchema - Postgres db schema

configTable - Table to store parameter configurations

speedResultTable - Table to store the speed results of experiments


****
**The root folder is "ml_template" and all the commands we are going to specify are referred to this root folder**


## Installation

Install the dependency using the following command:

```bash
pip install -r requirements.txt
```

* configargparse
* psycopg2
* torch
* sqlalchemy
* scipy
* pyyaml
* tensorboard
* tensorflow
* statsmodels
* torchmetrics
* tables
* sklearn



## Data Preparation

The traffic data files for Los Angeles (METR-LA) and the Bay Area (PEMS-BAY) are put into the `data/` folder. They are provided by [DCRNN](https://github.com/chnsh/DCRNN_PyTorch).

Run the following commands to generate train/test/val dataset at  `data/{METR-LA,PEMS-BAY}/{train,val,test}.npz`.
```bash
# Unzip the datasets
unzip data/metr-la.h5.zip -d data/
unzip data/pems-bay.h5.zip -d data/

# PEMS and METR-LA
python main.py -d defaults.ini -c config.ini -oDP 1 -dN METR-LA PEMS-BAY

```


## Train Model

When you train the model, you can run:

```bash
# Use METR-LA dataset
python main.py -d defaults.ini -c config.ini -dN METR-LA -pCC 1 -gP 1 -pO 1 -mN HA MA VAR ST_Norm Graph_WaveNet

# Use PEMS-BAY dataset
python main.py -d defaults.ini -c config.ini -dN PEMS-BAY -pCC 1 -gP 1 -pO 1 -mN HA MA VAR ST_Norm Graph_WaveNet
```

Hyper-parameters can be modified in config.ini file or from command line.


## Acknowledgments

[DCRNN-PyTorch](https://github.com/chnsh/DCRNN_PyTorch), [GCN](https://github.com/tkipf/gcn), [NRI](https://github.com/ethanfetaya/NRI) and [LDS-GNN](https://github.com/lucfra/LDS-GNN).

**Nicolas Tempelmeier**, for the evaluation template
