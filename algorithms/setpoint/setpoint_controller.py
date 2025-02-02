import math
class SetpointController():
    '''
    Setpoint controller for CO2 Fan is a hystreresis controller that turns on the fan on full speed when the CO2 concentration
    is above 700 ppm and turns off the fan when the CO2 concentration is below 600 ppm.
    '''
    # mapping = {
    # Winter actions 
    #     16 : [20, 22, 0.75, 0.0],
    #     18 : [20, 22, 0.75, 0.75],

    # Summer actions
    #     40 : [24, 26, 0.75, 0.0],
    #     42 : [24, 26, 0.75, 0.75],

    #     48: [5,50,0.0,0.0],  # Off action
    #     50: [5,50,0.0,0.75], Hvac off co2 on
    # }
    def __init__(self):
        self.is_co2_open = False
        self.is_hvac_open = False
        self.summer_hvac_on_co2_off = 40
        self.summer_hvac_on_co2_on = 42
        self.winter_hvac_on_co2_off = 16
        self.winter_hvac_on_co2_on = 18
        self.hvac_off_co2_on = 50
        self.off_action = 48
        self.summer_limits = [23,26]
        self.winter_limits = [20,23.5]

    def select_action(self, state):
        '''
        Act method for the controller
        '''
        co2 = state[0][10]
        temp = state[0][7]

        month = state[0][0]
        
        occupancy = state[0][9]
        # Determine season (summer or winter)
        is_summer = 6 <= month <= 9
        # Select seasonal limits based on the current season
        lower_limit, upper_limit = (
            self.summer_limits if is_summer else self.winter_limits
        )
        
        # Add hysteresis margins
        lower_hysteresis = lower_limit - 0.5
        upper_hysteresis = upper_limit + 0.5
        # Determine HVAC state based on the current temperature
        if not self.is_hvac_open:
            # If HVAC is off, decide to turn it on
            if temp < lower_limit:  # Too cold, need heating
                self.is_hvac_open = True
                self.is_cooling = False  # Enter heating mode
            elif temp > upper_limit:  # Too hot, need cooling
                self.is_hvac_open = True
                self.is_cooling = True  # Enter cooling mode
        else:
            # If HVAC is already on, use hysteresis to decide when to turn it off
            if self.is_cooling and temp <= lower_hysteresis:  # Cooling complete
                self.is_hvac_open = False
            elif not self.is_cooling and temp >= upper_hysteresis:  # Heating complete
                self.is_hvac_open = False
        # Handle CO2 Fan hysteresis
        if co2 > 800:
            self.is_co2_open = True
        elif co2 < 700:
            self.is_co2_open = False

        if occupancy > 0:
            if self.is_co2_open:
                return self.summer_hvac_on_co2_on if is_summer else self.winter_hvac_on_co2_on
            else:
                return self.summer_hvac_on_co2_off if is_summer else self.winter_hvac_on_co2_off
                
        else:
            #If not in working hours, close everything
            return self.off_action