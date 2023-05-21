# Patronus-Tool-Comparison
## CS520 Project

Group members: Chaithra Bhat, Dong Hyub Kim, Maryam Akbarpour, Saloni Chalkapurkar

Our project aims to compare and evaluate the two platforms, Neptune.ai and WandB (Weights and Biases), based on various evaluation criteria.

<br>

## Get Started with WandB

1. Create an account and install WandB

    To begin, it is important to set up an account and install the W&B (Weights & Biases) tools:
    - Create a free account by visiting https://wandb.ai/site and proceed to log in to your W&B account.
    - Install the W&B library on your local machine in a Python 3 environment using `pip`.
        - `pip install wandb`
2. Log in to W&B

    Next, import the W&B Python SDK and log in:
    - `wandb.login()`
    - Provide your API key when prompted.
3. Start a run and track hyperparameters

    - Initialize a W&B Run object in your Python script or notebook with `wandb.init()`. A single unit of computation logged by W&B is called a Run.
    - Pass a dictionary to the `config` parameter with key-value pairs of hyperparameter names and values. It captures any configurations that you want to associate with a specific experiment. For example:

    ```
    run = wandb.init(
    # Set the project where this run will be logged
    project="my-project",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 1e-3,
        "epochs": 10,
        "batch_size": 128,
        "dropout": random.uniform(0.01, 0.80)
    })
    ```
4. Log train metrics to WandB, to stream metrics during training of your ML model. For example:

    ```
    wandb.log({
        "train/train_loss": train_loss,
        "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
        "train/example_ct": example_ct
    })
    ```
5. Run your python script and navigate to the W&B App at https://wandb.ai/home to view the metrics you logged.

<br>

## Get Started with Neptune.ai

1. Sign up at https://neptune.ai/register
2. Create a new project in the Neptune workspace for storing your metadata
3. Install the Neptune client library on your local machine in a Python 3 environment using 
    - `pip install neptune`
4. Setting up authentication
    - From the API token dialog in Neptune, copy the export command and append the line to your .profile or other shell initialization file.
5. Additionally, you can also save the project name to the NEPTUNE_PROJECT environment variable in your system.
6. Add Neptune to your code

    Now we import Neptune in our code, initialize a run, and start logging. Create a Python script and enter the following commands:

    ```
    import neptune
    run = neptune.init_run(
        project="your-workspace-name/your-project-name",
        api_token="YourNeptuneApiToken", 
    )
    ```
    In the code above, we import the Neptune client library and initialize a run object. The run object automatically logs some system information and hardware consumption.
7. Log hyperparameters, metadata, and any configurations needed
    
    Now that the run is active, we will log the metadata. It is periodically synchronized with the Neptune servers in the background.

    Define some hyperparameters to track for the experiment and log them to the run object. For example:

    ```
    parameters = {
    "dense_units": 128,
    "activation": "relu",
    "dropout": 0.23,
    "learning_rate": 0.15,
    "batch_size": 64,
    "n_epochs": 30,
    }
    run["model/parameters"] = parameters
    ```
8. Track the training process by logging your training metrics and run your ML model. Also, log the evaluation results too.
9. Stop the connection and synchronize the data with the Neptune servers using `run.stop()`
10. Run your script and follow the link to explore the results and metadata in Neptune under the "Run details" view