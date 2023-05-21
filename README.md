# Patronus-Tool-Comparison
## CS520 Project

Group members: Chaithra Bhat, Dong Hyub Kim, Maryam Akbarpour, Saloni Chalkapurkar

Our project aims to compare and evaluate the two platforms, Neptune.ai and WandB (Weights and Biases), based on various evaluation criteria.

<br>

### Get Started with WandB

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
    - Pass a dictionary to the `config` parameter with key-value pairs of hyperparameter names and values. It captures any configurations that you want to associate with a specific experiment.

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

### Get Started with Neptune.ai