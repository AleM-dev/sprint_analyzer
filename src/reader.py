import numpy as np
import cv2
import yaml

class Reader:
    @staticmethod
    def read_yaml():
        try:
            yaml_file = open('ppt.yaml')
            data = yaml.load(yaml_file, yaml.SafeLoader)
            return data
        except yaml.YAMLError:
            print ("Error while parsing YAML file.")
            quit()
