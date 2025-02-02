
import math
class OnOffController():
    '''
    OnOffController class opens the fan when there is an occupation, else closes.
    '''
    # mapping = {
    # Winter actions 
    #     16 : [20, 22, 0.75, 0.0],
    #     18 : [20, 22, 0.75, 0.75],

    # Summer actions
    #     40 : [24, 26, 0.75, 0.0],
    #     42 : [24, 26, 0.75, 0.75],

    #     48: [5,50,0.0,0.0],  # Off action
    # }

    def __init__(self):
        self.is_open = False
        self.summer_hvac_on_co2_off = 40
        self.summer_hvac_on_co2_on = 42
        self.winter_hvac_on_co2_off = 16
        self.winter_hvac_on_co2_on = 18

        self.off_action = 48
        #Hvac on -off co2 on 
    def select_action(self, state):
        '''
        Act method for the controller
        '''
        occupancy = state[0][9]
        month = state[0][0]
        #Summer
        if occupancy > 0:

            if month >= 6 and month <= 9:
                return self.summer_hvac_on_co2_on
            else:
                return self.winter_hvac_on_co2_on
        else:
            return self.off_action