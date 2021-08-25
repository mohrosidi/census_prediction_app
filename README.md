# Salary prediction using US Census Data Using Random Forest Classifier with a Web API 

This project is an implementation of learning in Udacity Machine Learning for DevOps Engineers. This project focuses on implementing learning related to data version control (DVC), continuous integration using github and continuous deployment using heroku.

In this project, participants were asked to create a salary prediction model based on demographic data. For more information regarding the generated models and their terms of use, check out [model_card.md](https://github.com/mohrosidi/mlops_census/blob/master/model_card.md).

## Introduction

This repo holds the code to:
1. Clean and process raw census data
2. Train a Random Forest Classifier on a the processed data
3. Saves the model, label binarizer, encoder and datasets to a remote `dvc` repo
4. Deploy the model to Heroku

## Environment Set up

1. Download and install conda if you don’t have it already.
2. Install git either through conda (“`conda install git`”) or through your CLI, e.g. `sudo apt-get git`.
3. Clone this repo

```bash
git clone https://github.com/mohrosidi/mlops_census.git
```

4. create conda environment

```bash
conda create --name <env-name> python=3.8
conda activate <env-name>
```

5. install python modules that listed in `requiremnts.txt`

```bash
pip install -r requirements.txt
```

6. Setup [dvc](#setup-dvc)


## Set up S3

1. In your CLI environment install the<a href="https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html" target="_blank"> AWS CLI tool</a>.
2. Navigate to [AWS](https://aws.amazon.com/) and sign in to your AWS account if you have it
3. From the Services drop down select S3 and then click Create bucket.
4. Give your bucket a name, the rest of the options can remain at their default.

To use your new S3 bucket from the AWS CLI you will need to create an IAM user with the appropriate permissions. The full instructions can be found <a href="https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html#id_users_create_console" target="_blank">here</a>, what follows is a paraphrasing:

1. Sign in to the IAM console <a href="https://console.aws.amazon.com/iam/" target="_blank">here</a> or from the Services drop down on the upper navigation bar.
2. In the left navigation bar select **Users**, then choose **Add user**.
3. Give the user a name and select **Programmatic access**.
4. In the permissions selector, search for S3 and give it **AmazonS3FullAccess**
5. Tags are optional and can be skipped.
6. After reviewing your choices, click create user. 
7. Configure your AWS CLI to use the Access key ID and Secret Access key.

```bash
aws configure
```

## Data

Download [census.csv](https://raw.githubusercontent.com/udacity/nd0821-c3-starter-code/master/starter/data/census.csv) and copy it to data folder.
   * Information on the dataset can be found <a href="https://archive.ics.uci.edu/ml/datasets/census+income" target="_blank">here</a>.

## DVC Setup
This project uses `dvc` for artifact version control. How to setup `dvc` on your own environment and S3 bucket?

1. Create new repository on your github
2. Clone your new repository to your directory
3. Setup dvc to your system

```bash
dvc init
```

4. Add your files to track using dvc.

```bash
dvc add ./path/to/your/files
```

For this project you must track the data and model artifact

5. Add S3 remote repository to your dvc

```bash
dvc remote add -d <remote-name> s3://bucket/folder
```
6. Add your tracked files to .gitignore so your original files and artifacts won't be uploaded to github

7. Commit your changes to github
8. Push your commit to github
9. Finally, push your data and your model artifact to remote repository

```
dvc push --remote <remote-name>
```

The dvc remote location is found in the file `.dvc/config` - amend as necessary. 

**Important Note**: both Github Actions and Heroku requires the artifacts to be pulled from S3 via `dvc pull`. Be sure to include the AWS credentials and dvc setup for both build environments.

## Model

For simplicity, you can run the training process using this command:

```
python  starter/train_model.py
```
There are 4 files generate from this process:

1. `clean_census.csv` in data folder
2. model artifact in model folder

## GitHub Actions

In this project i have already setup testing process in `main.yml` (`./.github/workflows/main.yml`). If you want to run your own testing process using github action, you can change several part of that file.

1. Bucket location
2. AWS credential (you can create secret parameter in your owan repository)

## API Creation

* Create a RESTful API using FastAPI this must implement:
   * GET on the root giving a welcome message.
   * POST that does model inference.
   * Type hinting must be used.
   * Use a Pydantic model to ingest the body from POST. This model should contain an example.
    * Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
* Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

## API Deployment

This project uses Github actions (`.github/workflows`) for CI and Heroku for CD.

### Buildpacks required:
1. `heroku/python`
2. `heroku-community/apt`   - required to install apt packages (like `dvc`)

### Files required:
1. `Procfile`       - provides Heroku with the dyno startup commands
2. `Aptfile`        - apt dependencies to install. Here, it's the `dvc` version.
3. `runtime.txt`    - the Python runtime version associated to Heroku's buildpack

### Config Variables required:
1. `AWS_ACCESS_KEY_ID`
2. `AWS_SECRET_ACCESS_KEY`

These env variables are required as the dvc remote is in a S3 bucket.

## Acknowledgments

Thanks to [ashrielbrian](https://github.com/ashrielbrian/MLDevOps_census) who teach me how to wrap model object to class. Please see [model.py](https://github.com/mohrosidi/census_prediction_app/blob/master/starter/ml/model.py)
