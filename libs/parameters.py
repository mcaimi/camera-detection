#!/usr/bin/env python

try:
    import yaml
except Exception as e:
    raise e


# dictionary class that holds parameters
# load values from a yaml file
class Parameters(object):
    def __init__(self, data: dict):
        if type(data) is not dict:
            raise TypeError(f"Parameters: expected 'dict', got {type(data)}.")
        else:
            self.data = data

        for k in self.data.keys():
            if type(self.data.get(k)) != dict:
                self.__setattr__(k, self.data.get(k))
            else:
                self.__setattr__(k, Parameters(self.data.get(k)))


# loads a parameters file
def loadConfig(configPath: str) -> Parameters:
    try:
        with open(configPath) as parms:
            config_parms = yaml.safe_load(parms)
        parms: Parameters = Parameters(config_parms)
    except Exception as e:
        raise e

    # return data
    return parms
