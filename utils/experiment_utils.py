import os
import math
from datetime import datetime
import numpy as np
import torch
import sinergym
from sinergym.utils.constants import *
import inflect

def calculate_mean_and_std(all_obs_dict):
    means = []
    stds = []
    for key, values in all_obs_dict.items():
        if key == 'time_labels':
            # If 'time_labels' should be handled differently, you can log or process them separately
            means.append(None)  # Placeholder for time_labels as it's not numeric
            stds.append(None)  # Placeholder for time_labels as it's not numeric
            continue  # Skip to the next key
        means.append(np.mean(values))
        stds.append(np.std(values))
    return means,stds

def pluralize_keys(obs_dict):
    p = inflect.engine()
    return {p.plural(key): value for key, value in obs_dict.items()}

def update_combined_dict(all_obs_dict, combined_dict):
    # Merge all_obs_dict into combined_dict
    for key, value in all_obs_dict.items():
        if key not in combined_dict:
            combined_dict[key] = []
        combined_dict[key].extend(value)
    
    return combined_dict
def add_observation(all_obs_dict, obs_dict):
    obs_dict = pluralize_keys(obs_dict)
    for key, value in obs_dict.items():
        if key not in all_obs_dict:
            all_obs_dict[key] = []
        all_obs_dict[key].append(value)
    return all_obs_dict

def append_info_and_time_to_dict(observation,info,current_step, timesteps_per_hour):
    month, day,hour = int(observation['month']), int(observation['day_of_month']), int(observation['hour'])
    minute = int((current_step % timesteps_per_hour) * (60 / timesteps_per_hour))
    time_label = f"{month:02}-{day:02} {hour:02}:{minute:02}"

    observation['temp_violation'] = info['is_comfort_violated']
    observation['co2_violation'] = info['is_co2_violated']
    observation['time_label'] = time_label
    return observation
def append_fan_speed_to_dict(observation, window_fan_speed, ac_fan_speed):

    observation['window_fan_speed'] = np.float32(window_fan_speed)
    observation['ac_fan_speed'] = np.float32(ac_fan_speed)
    return observation

def append_fan_speed_to_observation(env,observation,action):
    obs_dict = dict(zip(env.get_wrapper_attr('observation_variables'), observation))
    obs_dict = append_fan_speed_to_dict(obs_dict, DEFAULT_A403V3_DISCRETE_FUNCTION(action)[3], DEFAULT_A403V3_DISCRETE_FUNCTION(action)[2]) 
    # Convert obs_dict back into data
    observation_variables = env.get_wrapper_attr('observation_variables') + ['window_fan_speed', 'ac_fan_speed']
    # Step 2: Extract the data values in the same order
    observation = [obs_dict[var] for var in observation_variables]
    return observation
# Means: [np.float32(8.666667), np.float32(16.169794), np.float32(11.507992), np.float32(28.934195), np.float32(37.565823), 
#         np.float32(24.0), np.float32(26.0), np.float32(26.5989), np.float32(39.966743), np.float32(3.105629), np.float32(521.407), np.float32(19820.014), 
#         np.float32(261332.2), np.float64(0.39448691220755155), np.float64(0.1938846421125782), None, np.float32(0.7494788), np.float32(0.7494788)]
# Standard Deviations: [np.float32(0.47140455), np.float32(5.3351545), np.float32(6.9179473), np.float32(5.099475), np.float32(16.559437),
#                        np.float32(0.0), np.float32(0.0), np.float32(2.2118723), np.float32(12.816699), np.float32(3.6032803), np.float32(136.5423), 
#                        np.float32(2086.485), np.float32(194333.78), np.float64(0.48874020532845774), np.float64(0.39533958524976426), 
                    #    None, np.float32(0.019764235), np.float32(0.019764235)]
                    
# {'month': np.float32(7.0), 'day_of_month': np.float32(10.0), 'hour': np.float32(0.0),
#  'outdoor_temperature': np.float32(28.666666), 'outdoor_humidity': np.float32(36.666668), '
# htg_setpoint': np.float32(4.13), 'clg_setpoint': np.float32(50.0), 'air_temperature': np.float32(26.72595), 
# 'air_humidity': np.float32(40.54236), 'people_occupant': np.float32(0.0), 'air_co2': np.float32(456.72827), 
# 'window_fan_energy': np.float32(0.0), 'total_electricity_HVAC': np.float32(0.0)}

obs_means = [np.float32(6.421548), np.float32(16.024471), np.float32(11.507992), np.float32(22.037941),
        np.float32(34.10978), np.float32(14.389556), np.float32(35.018764), np.float32(22.971294),
        np.float32(30.68262), np.float32(3.1064727), np.float32(587.5193), np.float32(5534.7583), 
        np.float32(149596.14), np.float64(0.25218405638836494), np.float64(0.06706045865184156)]

obs_stds = [np.float32(3.435255), np.float32(8.530136), np.float32(6.9179473), 
                      np.float32(9.132451), np.float32(23.794716), np.float32(8.339892), 
                      np.float32(13.180379), np.float32(4.2705617), np.float32(17.075714), 
                      np.float32(3.6196496), np.float32(164.96347), np.float32(8914.488), 
                      np.float32(190947.97), np.float64(0.43426634464562747), np.float64(0.2501266749813906)]
# obs_means =  [-1.88616163e-01 ,-2.57461701e-01 , 2.59861111e-01, -2.36515783e-05,
#  -1.79738013e-04  ,2.37658727e+01  ,3.70050000e+01,  2.21512500e+01,
#   2.51512500e+01 , 2.34620032e+01 , 3.42801016e+01 , 1.71827337e+00,
#   7.50636800e+02 , 2.44119660e-01  ,1.02112024e+01 , 4.52476931e+02,
#   4.37303756e+05]
# obs_stds =  [5.76098289e-01 ,7.52494634e-01, 4.38558222e-01, 7.07073908e-01,
#  7.07139627e-01, 8.30790585e+00 ,2.19055985e+01 ,1.45889459e+00,
#  1.45889459e+00 ,1.74003051e+00 ,1.61345113e+01 ,1.99147147e+00,
#  6.55276029e+02 ,4.34552292e-01 ,5.27910243e+00 ,6.01012338e+01,
#  3.89152352e+05]
def normalize_observation(observation, means, stds):
    """
    Normalize a raw observation using the provided means and standard deviations.
    
    Parameters:
        observation (array-like): Raw observation to normalize.
        means (array-like): List or array of mean values for each feature.
        stds (array-like): List or array of standard deviation values for each feature.
        
    Returns:
        np.ndarray: Normalized observation where each feature is scaled as (x - mean) / std.
    """
    if isinstance(observation, torch.Tensor):
        # Move to CPU if needed and convert to NumPy
        observation = observation.cpu().numpy()
    observation = np.array(observation)  # Ensure observation is a numpy array
    means = np.array(means)
    stds = np.array(stds)
    
    # Prevent division by zero for features with zero standard deviation
    stds[stds == 0] = 1.0
    
    normalized_obs = (observation - means) / stds
    return normalized_obs

def create_experiment_name(env_name, episodes,algorithm_name):
    experiment_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
    experiment_name = algorithm_name+'-' + env_name + \
        '-episodes-' + str(episodes)
    experiment_name += '_' + experiment_date
    return experiment_name

def save_run_metrics(save_dir,avg_Train_reward, avg_Train_power, avg_total_co2_concentration,avg_occupnacy_co2_concentration):
    with open(os.path.join(save_dir, "run_metrics.txt"), "a") as f:
        f.write(f"Train Average Reward = {avg_Train_reward}\n")
        f.write(f"Train Average Power = {avg_Train_power}\n")
        f.write(f"Train Average Total CO2 Concentration = {avg_total_co2_concentration}\n")
        f.write(f"Train Average Occupancy CO2 Concentration = {avg_occupnacy_co2_concentration}\n")

def calculate_time_label(month_sin, month_cos, hour_sin, hour_cos, current_step, timesteps_per_hour):
    """
    Calculate the human-readable time label from sinusoidal time features and step information.

    Args:
        month_sin (float): Sine of the month.
        month_cos (float): Cosine of the month.
        hour_sin (float): Sine of the hour.
        hour_cos (float): Cosine of the hour.
        is_weekend (int): 1 if weekend, 0 otherwise.
        current_step (int): Current timestep index.
        timesteps_per_hour (int): Number of timesteps per hour.

    Returns:
        str: Formatted time label.
    """
    # Reconstruct the month (1-12)
    month_angle = math.atan2(month_sin, month_cos)
    month = int(((month_angle + 2 * math.pi) % (2 * math.pi)) * (12 / (2 * math.pi)) + 1)

    # Reconstruct the hour (0-23)
    hour_angle = math.atan2(hour_sin, hour_cos)
    hour = int(((hour_angle + 2 * math.pi) % (2 * math.pi)) * (24 / (2 * math.pi)))

    # Calculate the minute
    minute = int((current_step % timesteps_per_hour) * (60 / timesteps_per_hour))

    time_label = f"Month: {month:02}, Hour: {hour:02}:{minute:02}"
    return time_label