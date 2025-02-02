import gymnasium as gym
import pandas as pd
import sinergym
import numpy as np
from environments.reward import *
from environments.environment import create_environment
from environments.environment import CO2_AND_TEMP_REWARD_CONFIG
from utils.experiment_utils import *
from algorithms.onoff.on_off_controller import *
from algorithms.setpoint.setpoint_controller import *
from tqdm import tqdm
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




timesteps_per_hour = 12  # 5 min interval
timestep_per_day = timesteps_per_hour * 24
total_days = 365
total_steps = timestep_per_day * total_days
# {'month': np.float32(7.0), 'day_of_month': np.float32(10.0), 'hour': np.float32(0.0),
#  'outdoor_temperature': np.float32(28.666666), 'outdoor_humidity': np.float32(36.666668), '
# htg_setpoint': np.float32(4.13), 'clg_setpoint': np.float32(50.0), 'air_temperature': np.float32(26.72595), 
# 'air_humidity': np.float32(40.54236), 'people_occupant': np.float32(0.0), 'air_co2': np.float32(456.72827), 
# 'window_fan_energy': np.float32(0.0), 'total_electricity_HVAC': np.float32(0.0)}
def run_simulation(env_id, controller_name, controller):
    # Column names for the DataFrame
    # Time values first, then outdoor values. Then, the rest of the observation values
    column_names_X = ['month', 'day_of_month', 'hour', 'minute', 'outdoor_temperature', 
                      'outdoor_humidity', 'air_temperature', 'air_humidity', 
                      'people_occupant', 'air_co2', 'window_fan_energy', 
                      'total_electricity_HVAC', 'heating_setpoint', 
                      'cooling_setpoint', 'ac_fan_speed', 'window_fan_speed']
    # Create the DataFrame to store the data
    df = pd.DataFrame(columns=column_names_X)
    
    env = create_environment(
        datetime(1997, 1, 1), datetime(1997, 12, 31),
        CO2andTemperatureReward, env_id, timesteps_per_hour,
        CO2_AND_TEMP_REWARD_CONFIG
    )
    
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    current_step = 1
    data_rows= []
    
    with tqdm(total=total_steps, desc=f"{controller_name} - {env_id}", ncols=120, unit="chunk", leave=True) as pbar:
        while current_step < total_steps:
            action = controller.select_action(state)
            observation, reward, truncated, terminated, info = env.step(action)
            done = terminated or truncated
            # Extract relevant features from observation
            month, day, hour = int(observation[0]), int(observation[1]), int(observation[2])  # Assuming these are indices 0, 1, and 2
            minute = int((current_step % timesteps_per_hour) * (60 / timesteps_per_hour))
            
            # Store state, action, next state
            action_array = DEFAULT_A403V3_DISCRETE_FUNCTION(action)
            # Discard 5th and 6th elements of observation
            filtered_observation = np.delete(observation, [5, 6])
            #Also discard first 3 elements of observation
            filtered_observation = np.delete(filtered_observation, [0, 1, 2])
            
            # Create the data row (state + action + timestamp)
            row = np.concatenate([ [month,day,hour,minute], filtered_observation, action_array ])
            # Append the row to the DataFrame
            data_rows.append(row)
            state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            if done:
                env.reset()
            
            current_step += 1
            
            if current_step % 100 == 0:
                pbar.set_postfix_str(f"Timestep {pbar.n + 1}/{pbar.total}")
                pbar.update(100)
    
    env.close()
    df = pd.DataFrame(data_rows, columns=column_names_X)
    return df

# Run all experiments and store data
all_data = []
# Define environment IDs and controllers
A403_ENV_IDS = [
    "Eplus-A403v3-hot-discrete-v1",
    "Eplus-A403v3-mixed-discrete-v1",
    "Eplus-A403v3-cool-discrete-v1"
]

controllers = {
    "Setpoint": SetpointController(),
    "OnOff": OnOffController()
}
# Loop through each env_id and controller_name, run the simulation, and save to CSV
i = 0
for env_id in A403_ENV_IDS:
    for controller_name, controller in controllers.items():
        data = run_simulation(env_id, controller_name, controller)
        
        # Save the DataFrame to CSV with a dynamic filename
        csv_filename = f"simulation_data_{i+1}.csv"  # File name includes the index (i+1)
        data.to_csv(csv_filename, index=False)
        print(f"Saved simulation data for {controller_name} - {env_id} to {csv_filename}")
        i+=1

