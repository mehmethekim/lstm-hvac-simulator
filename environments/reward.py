import gymnasium as gym
import sinergym
import matplotlib.pyplot as plt
import sinergym
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from sinergym.utils.rewards import LinearReward
from typing import Any, Dict, List, Tuple, Union
from sinergym.utils.constants import LOG_REWARD_LEVEL, YEAR
from datetime import datetime
import math
#This class combines Co2Reward and the classic reward function
class CO2andTemperatureReward(LinearReward):
    def __init__(
            self,
            co2_variable: str,
            energy_variables: List[str],
            temperature_variables: List[str],
            range_comfort_winter: Tuple[int, int],
            range_comfort_summer: Tuple[int, int],
            energy_weight: float = 0.3,
            co2_weight: float = 0.3,
            temperature_weight: float = 0.3,
            summer_start: Tuple[int, int] = (6, 1),
            summer_final: Tuple[int, int] = (9, 30),
            lambda_energy: float = 1e-2,
            lambda_co2: float = 1.0,
            lambda_temperature: float = 1.0,
            co2_threshold: float = 700,

        ):
            super(LinearReward, self).__init__()
            self.co2_variable = co2_variable
            self.energy_names = energy_variables
            self.temp_names = temperature_variables
            self.W_energy = energy_weight
            self.W_co2 = co2_weight
            self.W_temperature = temperature_weight
            self.range_comfort_winter = range_comfort_winter
            self.range_comfort_summer = range_comfort_summer
            self.summer_start = summer_start
            self.summer_final = summer_final

            self.lambda_energy = lambda_energy
            self.lambda_co2 = lambda_co2
            self.lambda_temperature = lambda_temperature
            self.co2_threshold = co2_threshold

            self.energy_rew_arr = []
            self.co2_rew_arr = []
            self.daily_timestep_count = 0
            self.timesteps_per_day = 288


    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the reward function.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value).

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """

        try:
            assert all(temp_name in list(obs_dict.keys())
                       for temp_name in self.temp_names)
        except AssertionError as err:
            self.logger.error(
                'Some of the temperature variables specified are not present in observation.')
            raise err
        try:
            assert all(energy_name in list(obs_dict.keys())
                       for energy_name in self.energy_names)
        except AssertionError as err:
            self.logger.error(
                'Some of the energy variables specified are not present in observation.')
            raise err
        try:
            assert self.co2_variable in list(obs_dict.keys())
        except AssertionError as err:
            self.logger.error(
                'CO2 variable specified is not present in observation.')
            raise err
        # Energy penalty
        energy_consumed, energy_values = self._get_energy_consumed(obs_dict)
        energy_penalty = self._get_energy_penalty(energy_values)

        # CO2 penalty
        co2_concentration = obs_dict[self.co2_variable]
        co2_reward = self._get_co2_reward(co2_concentration)

         # Comfort violation calculation
        temp_reward, temp_violations = self._get_temperature_violation(obs_dict)

        # Weighted sum of both terms
        reward, energy_term ,co2_term,comfort_term = self._get_reward(energy_penalty, co2_reward, temp_reward)

        reward_terms = {
            'energy_term': energy_term,
            'co2_term': co2_term,
            'comfort_term': comfort_term,
            'energy_weight': self.W_energy,
            'co2_weight': self.W_co2,
            'temperature_weight': self.W_temperature,
            'abs_energy_penalty': energy_penalty,
            'abs_comfort_penalty': temp_violations,
            'abs_co2_penalty': co2_reward,
            'total_power_demand': energy_consumed,
            'total_temperature_violation': temp_reward,
            'co2_concentration': co2_concentration,
            'is_co2_violated': obs_dict['people_occupant'] > 0 and co2_concentration > self.co2_threshold, 
            'is_comfort_violated': obs_dict['people_occupant'] > 0 and temp_reward < 0,
            'is_occupied': obs_dict['people_occupant'] > 0
        }
        return reward, reward_terms
    def _get_co2_reward(self, co2_concentration: float) -> float:
        """
        Calculate the penalty based on CO2 concentration.

        Args:
            co2_concentration (float): CO2 concentration in ppm.

        Returns:
            float: Negative absolute CO2 penalty.
        """
        
        if co2_concentration < self.co2_threshold:
            co2_reward = 1.0
        else:
            co2_reward = -20.0
        
        return co2_reward

    def _get_temperature_violation(self, obs_dict: Dict[str, Any]) -> Tuple[float, List[float]]:
        """
        Calculate the total temperature violation (ºC) in the current observation.

        Returns:
            Tuple[float, List[float]]: Total temperature violation (ºC) and list with temperature violation in each zone.
        """
        # Extract month and reconstruct day if necessary
        
        month = obs_dict['month']
        day = obs_dict['day_of_month']
        year = YEAR
        current_dt = datetime(int(year), int(month), int(day))

        # Periods
        summer_start_date = datetime(
            int(year),
            self.summer_start[0],
            self.summer_start[1])
        summer_final_date = datetime(
            int(year),
            self.summer_final[0],
            self.summer_final[1])

        if current_dt >= summer_start_date and current_dt <= summer_final_date:
            temp_range = self.range_comfort_summer
        else:
            temp_range = self.range_comfort_winter

        # Process temperature values
        temp_values = [v for k, v in obs_dict.items() if k in self.temp_names]
        total_temp_violation = 0.0
        temp_violations = []
        person_count = obs_dict['people_occupant']
        total_reward = 0
        for T in temp_values:
            if person_count == 0:
                # If no one is in the room, ignore the comfort violation
                break
            if T >= temp_range[0] and T <= temp_range[1]:
                # Inside comfort limits, add reward
                total_reward += 1
            else:
                # Outside comfort limits, add penalty
                temp_violation = min(abs(temp_range[0] - T), abs(T - temp_range[1]))
                total_reward -= 5*(1+temp_violation)
                
                temp_violations.append(temp_violation)
                total_temp_violation += temp_violation

        return total_reward, temp_violations
    def _get_reward(self, energy_penalty: float, co2_penalty: float,temperature_penalty: float) -> Tuple[float, float, float,float]:
        """
        Calculate the reward value using penalties for energy, CO2 and temperature.

        Args:
            energy_penalty (float): Negative absolute energy penalty value.
            co2_penalty (float): Negative absolute CO2 penalty value.
            temperature_penalty (float): Negative absolute temperature penalty value.

        Returns:
            Tuple[float, float, float,float]: Total reward, energy term, CO2 term, temperature term.
        """

        energy_term = self.lambda_energy * self.W_energy * energy_penalty
        co2_term = self.lambda_co2 * self.W_co2 * co2_penalty
        temperature_term = self.lambda_temperature * self.W_temperature * temperature_penalty
        reward = energy_term + co2_term + temperature_term
        self.energy_rew_arr.append(energy_term)
        self.co2_rew_arr.append(co2_term)


        # Increment daily timestep count
        self.daily_timestep_count += 1

        # Log and reset at the end of the day
        if self.daily_timestep_count == self.timesteps_per_day*9:
            avg_energy_reward = sum(self.energy_rew_arr) / len(self.energy_rew_arr)
            avg_co2_reward = sum(self.co2_rew_arr) / len(self.co2_rew_arr)
            # print(f"Lambdas: Energy: {self.lambda_energy}, CO₂: {self.lambda_co2}")
            # print(f"Energy Weight: {self.W_energy}")
            # print(f"Average Energy Reward for the Day: {avg_energy_reward}")
            # print(f"Average CO₂ Reward for the Day: {avg_co2_reward}")

            self.energy_rew_arr.clear()
            self.co2_rew_arr.clear()
            self.daily_timestep_count = 0
        
        return reward, energy_term, co2_term, temperature_term



class CO2Reward(LinearReward):
    def __init__(
        self,
        co2_variable: str,
        energy_variables: List[str],
        energy_weight: float = 0.3,
        lambda_energy: float = 1e-2,
        lambda_co2: float = 1.0,
        ideal_co2: float = 700,
    ):
        """
        Initialize the CO2Reward class.

        Args:
            co2_variable (str): Name of the CO2 concentration variable.
            energy_variables (List[str]): Names of energy-related variables.
            energy_weight (float): Weight for energy term in the reward.
            lambda_energy (float): Scaling factor for energy penalty.
            lambda_co2 (float): Scaling factor for CO2 penalty.
            ideal_co2 (float): Ideal CO2 level (ppm).
        """
        super(LinearReward, self).__init__()
        self.co2_variable = co2_variable
        self.energy_names = energy_variables
        self.W_energy = energy_weight
        self.lambda_energy = lambda_energy
        self.lambda_co2 = lambda_co2
        self.ideal_co2 = ideal_co2
        self.energy_rew_arr = []
        self.co2_rew_arr = []
        self.daily_timestep_count = 0
        self.timesteps_per_day = 288


    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the reward function.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value).

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """
        # Energy penalty
        energy_consumed, energy_values = self._get_energy_consumed(obs_dict)
        energy_penalty = self._get_energy_penalty(energy_values)

        # CO2 penalty
        co2_concentration = obs_dict[self.co2_variable]
        co2_reward = self._get_co2_reward(co2_concentration)

        # Weighted sum of terms
        reward, energy_term, co2_term = self._get_reward(energy_penalty, co2_reward)

        reward_terms = {
            'energy_term': energy_term,
            'co2_term': co2_term,
            'reward_weight': self.W_energy,
            'abs_energy_penalty': energy_penalty,
            'abs_co2_penalty': co2_reward,
            'total_power_demand': energy_consumed,
            'co2_concentration': co2_concentration
        }
        
        return reward, reward_terms

    def _get_co2_reward(self, co2_concentration: float) -> float:
        """
        Calculate the penalty based on CO2 concentration.

        Args:
            co2_concentration (float): CO2 concentration in ppm.

        Returns:
            float: Negative absolute CO2 penalty.
        """
        
        if co2_concentration < self.ideal_co2:
            co2_reward = 1.0
        else:
            co2_reward = -20.0
        
        return co2_reward

    def _get_reward(self, energy_penalty: float, co2_penalty: float) -> Tuple[float, float, float]:
        """
        Calculate the reward value using penalties for energy and CO2.

        Args:
            energy_penalty (float): Negative absolute energy penalty value.
            co2_penalty (float): Negative absolute CO2 penalty value.

        Returns:
            Tuple[float, float, float]: Total reward, energy term, CO2 term.
        """

        energy_term = self.lambda_energy * self.W_energy * energy_penalty
        co2_term = self.lambda_co2 * (1 - self.W_energy) * co2_penalty
        reward = energy_term + co2_term
        self.energy_rew_arr.append(energy_term)
        self.co2_rew_arr.append(co2_term)


        # Increment daily timestep count
        self.daily_timestep_count += 1

        # Log and reset at the end of the day
        if self.daily_timestep_count == self.timesteps_per_day*9:
            avg_energy_reward = sum(self.energy_rew_arr) / len(self.energy_rew_arr)
            avg_co2_reward = sum(self.co2_rew_arr) / len(self.co2_rew_arr)
            # print(f"Lambdas: Energy: {self.lambda_energy}, CO₂: {self.lambda_co2}")
            # print(f"Energy Weight: {self.W_energy}")
            # print(f"Average Energy Reward for the Day: {avg_energy_reward}")
            # print(f"Average CO₂ Reward for the Day: {avg_co2_reward}")

            self.energy_rew_arr.clear()
            self.co2_rew_arr.clear()
            self.daily_timestep_count = 0
        
        return reward, energy_term, co2_term
class MyCustomReward(LinearReward):
    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[int, int],
        range_comfort_summer: Tuple[int, int],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.3,
        lambda_energy: float = 1e-4,
        lambda_temperature: float = 1.0
    ):
        
        super(LinearReward, self).__init__()
        # Name of the variables
        self.temp_names = temperature_variables
        self.energy_names = energy_variables

        # Reward parameters
        self.range_comfort_winter = range_comfort_winter
        self.range_comfort_summer = range_comfort_summer
        self.W_energy = energy_weight
        self.lambda_energy = lambda_energy
        self.lambda_temp = lambda_temperature

        # Summer period
        self.summer_start = summer_start  # (month,day)
        self.summer_final = summer_final  # (month,day)

        self.logger.info('Reward function initialized.')
        self.comfort_term_arr = []
        self.energy_term_arr = []
    def __call__(self, obs_dict: Dict[str, Any]
                 ) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """
        # Check variables to calculate reward are available
        try:
            assert all(temp_name in list(obs_dict.keys())
                       for temp_name in self.temp_names)
        except AssertionError as err:
            self.logger.error(
                'Some of the temperature variables specified are not present in observation.')
            raise err
        try:
            assert all(energy_name in list(obs_dict.keys())
                       for energy_name in self.energy_names)
        except AssertionError as err:
            self.logger.error(
                'Some of the energy variables specified are not present in observation.')
            raise err
        # Get the number of people in the room (occupancy)
        people_occupant = obs_dict['people_occupant']
        # Energy calculation
        energy_consumed, energy_values = self._get_energy_consumed(obs_dict)
        energy_penalty = self._get_energy_penalty(energy_values,people_occupant)

        # Comfort violation calculation
        total_temp_violation, temp_violations = self._get_temperature_violation(
            obs_dict)
        comfort_penalty = self._get_comfort_penalty(temp_violations)

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward(
            energy_penalty, comfort_penalty)

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'reward_weight': self.W_energy,
            'abs_energy_penalty': energy_penalty,
            'abs_comfort_penalty': comfort_penalty,
            'total_power_demand': energy_consumed,
            'total_temperature_violation': total_temp_violation
        }
        
        return reward, reward_terms

    def _get_energy_consumed(self, obs_dict: Dict[str,
                                                  Any]) -> Tuple[float,
                                                                 List[float]]:
        """Calculate the total energy consumed in the current observation.

        Args:
            obs_dict (Dict[str, Any]): Environment observation.

        Returns:
            Tuple[float, List[float]]: Total energy consumed (sum of variables) and List with energy consumed in each energy variable.
        """

        energy_values = [
            v for k, v in obs_dict.items() if k in self.energy_names]

        # The total energy is the sum of energies
        total_energy = sum(energy_values)

        return total_energy, energy_values

    def _get_temperature_violation(
            self, obs_dict: Dict[str, Any]) -> Tuple[float, List[float]]:
        """Calculate the total temperature violation (ºC) in the current observation.

        Returns:
            Tuple[float, List[float]]: Total temperature violation (ºC) and list with temperature violation in each zone.
        """

        month = obs_dict['month']
        day = obs_dict['day_of_month']
        year = YEAR
        current_dt = datetime(int(year), int(month), int(day))

        # Periods
        summer_start_date = datetime(
            int(year),
            self.summer_start[0],
            self.summer_start[1])
        summer_final_date = datetime(
            int(year),
            self.summer_final[0],
            self.summer_final[1])

        if current_dt >= summer_start_date and current_dt <= summer_final_date:
            temp_range = self.range_comfort_summer
        else:
            temp_range = self.range_comfort_winter

        temp_values = [
            v for k, v in obs_dict.items() if k in self.temp_names]
        total_temp_violation = 0.0
        temp_violations = []
        person_count = obs_dict['people_occupant']
        for T in temp_values:
            if T < temp_range[0] or T > temp_range[1]:
                temp_violation = min(
                    abs(temp_range[0] - T), abs(T - temp_range[1]))
                if person_count==0 :
                    # if there is no person in the room, the comfort violation is not considered 
                    break
                temp_violations.append(temp_violation)
                total_temp_violation += temp_violation
        
        
        return total_temp_violation, temp_violations

    def _get_energy_penalty(self, energy_values: List[float],occupancy: int) -> float:
        """Calculate the negative absolute energy penalty based on energy values

        Args:
            energy_values (List[float]): Energy values

        Returns:
            float: Negative absolute energy penalty value
        """
        
        total_energy = sum(energy_values)
    
        if occupancy == 0:
            # Penalize energy consumption more heavily when the room is empty
            energy_penalty = -2*total_energy  # Double the penalty when no people are present
        else:
            # Normal energy penalty
            energy_penalty = -total_energy   
        return energy_penalty

    def _get_comfort_penalty(self, temp_violations: List[float]) -> float:
        """Calculate the negative absolute comfort penalty based on temperature violation values

        Args:
            temp_violations (List[float]): Temperature violation values

        Returns:
            float: Negative absolute comfort penalty value
        """
        comfort_penalty = -sum(temp_violations)
        return comfort_penalty

    def _get_reward(self, energy_penalty: float,
                    comfort_penalty: float) -> Tuple[float, float, float]:
        """It calculates reward value using the negative absolute comfort and energy penalty calculates previously.

        Args:
            energy_penalty (float): Negative absolute energy penalty value.
            comfort_penalty (float): Negative absolute comfort penalty value.

        Returns:
            Tuple[float,float,float]: total reward calculated, reward term for energy, reward term for comfort.
        """
        
        energy_term = self.lambda_energy * self.W_energy * energy_penalty
        comfort_term = self.lambda_temp * \
            (1 - self.W_energy) * comfort_penalty
            
        self.comfort_term_arr.append(comfort_term)
        self.energy_term_arr.append(energy_term)
        #print("comfort reward:",np.mean(self.comfort_term_arr),", energy term: ",np.mean(self.energy_term_arr))
        reward = energy_term + comfort_term
        # print("Total reward: ",reward, "Energy term: ",energy_term, "Comfort term: ",comfort_term)
        return reward, energy_term, comfort_term
