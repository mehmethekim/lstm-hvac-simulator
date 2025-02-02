import numpy as np
def calculate_next_state(observation, timestep=5):
    """
    Calculate the next observation by adding the timestep to time values.
    
    Parameters:
        observation (list or np.array): Contains [month, day_of_month, hour, minute, ...]
        timestep (int): Number of minutes to increment (default is 5 minutes)
    
    Returns:
        next_observation (np.array): Updated observation with incremented time values
    """
    # Extract time values
    month, day, hour, minute = int(observation[0]), int(observation[1]), int(observation[2]), int(observation[3])
    
    # Define days in each month (assuming no leap year)
    days_in_month = {
        1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 
        7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
    }
    
    # Update minute and handle overflow
    minute += timestep
    if minute >= 60:
        hour += minute // 60
        minute = minute % 60  # Keep only remaining minutes
    
    # Update hour and handle overflow
    if hour >= 24:
        day += hour // 24
        hour = hour % 24  # Keep only remaining hours
    
    # Update day and handle month overflow
    if day > days_in_month[month]:
        day = 1  # Reset day
        month += 1  # Move to next month
    
    # Update month and handle year overflow (assuming single-year simulation)
    if month > 12:
        month = 1  # Reset month to January
    
    # Create updated observation with new time values
    next_observation = np.copy(observation)
    next_observation[0], next_observation[1], next_observation[2], next_observation[3] = month, day, hour, minute
    
    return next_observation

# obs = np.array([1.0, 1.0, 22.0, 0.0])  # January 1st, 23:55
# next_obs = calculate_next_state(obs, 5)
# print(next_obs)  # Expected output: [1.0, 2.0, 0.0, 0.0] (January 2nd, 00:00)