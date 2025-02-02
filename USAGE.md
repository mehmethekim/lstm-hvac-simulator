
# Environment Creation
Virtual environment is used to setup the environment

First install virtual environment using the command
```
pip3 install virtualenv
```
Create a virtual environment using the command
```
python3 -m venv sinergym-env
```
Activate the virtual environment using the command
```
source sinergym-env/bin/activate
```
Install the required packages using the command
```
pip3 install -r requirements.txt
```

# Custom environment usage

To be able to use custom environments, you need to change some files

Inside the sinergym-env dictionary, go to sinergym-env/lib/sinergym/data/buildings and add your epJSON file

Then add your default json file to sinergym-env/lib/sinergym/data/default_configuration

Also you need to add your custom discretization process into sinergym-env/lib/sinergym/utils/constants.py


# SSH Screen

screeen -S my_session
ctrl+a d detach
screen -r my_session

screen -list 
screein -r id