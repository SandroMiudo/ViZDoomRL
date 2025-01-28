# ViZDoomRL

This project focusses on Reinforcement Learning using the ViZDoom enviroment from **[gymnasium][1]**.

## Run

### Auto Run

Install **conda** using following instructions provided **[here][2]**.

Clone this repo and execute the run script inside the **script** folder from the root directory.
 
```shell
git clone https://github.com/SandroMiudo/ViZDoomRL.git

cd VizDoomRL

source scripts/run.sh (optional DBG argument can be specified) 
```

The `source` command needs to be specified to execute the script within the current shell environment, rather than spawning a new subshell.

### Manual run

In order to have more control over how the parameters are set in the script, the alternative approach is to directly pass arguments to the python script.

This approach requires to manually create+activate the conda enviroment using this **[file][3]**.

```shell
conda env create --no-default-packages -f conda_env.yml
```

If you created the conda enviroment you can now invoke following :

```shell
python3 -m src.agent.RLSingleQNetwork [OPTIONS...] CNF
```

To gather the options which are available just exec with `-h`.

A basic run, in which the epsilon parameters and the learning rate is set manunally would look like this : 

```shell
python3 -m src.agent.RLSingleQNetwork --train --resX 240 --resY 320\
--epsilon 0.05 --epsilon-update 0.0001 --learning 1e-7
```

A basic run, in which only the delta parameter should be modified would look something like this :

```shell
python3 -m src.agent.RLSingleQNetwork --train --resX 240 --resY 320\
--delta 0.95 --learning 1e-7
```

[1]: https://vizdoom.farama.org/ "vizdoom"
[2]: https://conda.io/projects/conda/en/latest/user-guide/install/index.html "conda"
[3]: <README.md> "conda packages"