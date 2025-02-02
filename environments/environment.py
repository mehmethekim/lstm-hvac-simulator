import gymnasium as gym
from datetime import datetime, timedelta
from sinergym.utils.wrappers import DatetimeWrapper

REWARD_CONFIG = {
    'temperature_variables': ['air_temperature'],
    'energy_variables': ['total_electricity_HVAC'],
    'range_comfort_winter': (20.0, 23.5),
    'range_comfort_summer': (23.0, 26.0),
    'energy_weight':  0.3,
    'lambda_energy':  1e-4,
    'lambda_temperature':1.0,
}

CO2_REWARD_CONFIG = {
    'co2_variable': 'air_co2',
    'energy_variables': ['total_electricity_HVAC'],
    'energy_weight': 0.3,
    'lambda_energy': 1e-2,
    'lambda_co2': 1.0,
    'ideal_co2': 700,
}
CO2_AND_TEMP_REWARD_CONFIG = {
    'temperature_variables': ['air_temperature'],
    'co2_variable': 'air_co2',
    'energy_variables': ['total_electricity_HVAC'],
    'range_comfort_winter': (20.0, 23.5),
    'range_comfort_summer': (23.0, 26.0),
    'energy_weight': 0.3,
    'co2_weight': 0.3,
    'temperature_weight': 0.3,
    'lambda_energy': 1e-2,
    'lambda_temperature': 1.0,
    'lambda_co2': 1.0,
    'co2_threshold': 700,
}

# Helper to create a new environment with given start and end dates
def create_environment(start_date, end_date, reward_fn,env_name='Eplus-A403v3-hot-discrete-v1', timesteps_per_hour=12, reward_kwargs=REWARD_CONFIG):
    """
    Helper function to create a Gymnasium environment with specific configurations.

    Args:
        
        start_date (datetime): Start date for the environment simulation.
        end_date (datetime): End date for the environment simulation.
        reward_fn (class): Custom reward class for the environment.
        env_name (str): The name of the environment to create.
        timesteps_per_hour (int): Number of timesteps per hour (default: 12 for 5-minute intervals).
        reward_kwargs (dict, optional): Additional parameters for the reward function.

    Returns:
        gym.Env: Configured Gymnasium environment instance.
    """
    extra_params = {
        'timesteps_per_hour': timesteps_per_hour,
        'runperiod': (
            start_date.day, start_date.month, start_date.year,
            end_date.day, end_date.month, end_date.year
        )
    }

    env = gym.make(env_name, 
                   reward=reward_fn,
                   reward_kwargs=reward_kwargs,
                   config_params=extra_params)
    #env = DatetimeWrapper(env)
    return env