#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:14:49 2025

@author: murdism
"""


import json
import os
from collections import Counter

# mapping between intentions and actions 
pedestrian_map = {'Moving' : 'walking', 
                  'Crossing_road_from_left' : 'crossing', 
                  'Pushing_Object' : 'walking', 
                  'Crossing_road_from_right': 'crossing', 
                  'Moving_towards' : 'walking', 
                  'Breaking' : 'walking', 
                  'Moving_away' : 'walking', 
                  'Turning_left' : 'walking', 
                  'Waiting_to_cross' : 'waiting_to_cross', 
                  'Crossing' : 'crossing', 
                  'Stopped' : 'stopped'}

# we keep on the pavements 
cyclist_map = {'Moving' : 'lane-keeping',
               'Moving_towards' : 'lane-keeping',
               'Turning_left' : 'turn-left',
               'Waiting_to_cross' : 'waiting_to_cross',
               'Crossing' : 'crossing'} 

# For motorbikes
motorbike_ignore = set([('Moving', 'On_pavement'), 
                        ('Moving_towards', 'On_left_pavement'), 
                        ('Moving_towards', 'On_right_pavement')]) 
motorbike_map = {'Stopped' : 'stopped',
                  'Breaking' : 'braking',
                  'Moving_left' : 'merge-left',
                  'Moving_right' : 'merge-right',
                  'Turning_left' : 'turn-left',
                  'Turning_right' : 'turn-right',
                  'Moving' : 'lane-keeping',
                  'Moving_towards' : 'lane-keeping',
                  'Moving_away' : 'lane-keeping',
                  'Indicating_left' : 'lane-keeping'}

# For small motorised vehicles
small_motorised_vehicles_ignore = set([('Moving_towards', 'On_left_pavement'),
                                       ('Moving_towards', 'On_right_pavement'),
                                       ('Moving', 'On_left_pavement'), 
                                       ('Moving', 'On_right_pavement')]) 
small_motorised_vehicles_map = {'Crossing_road_from_right': 'crossing', 
                                'Crossing' : 'crossing',
                                'Waiting_to_cross' : 'waiting_to_cross',
                                'Stopped' : 'stopped',
                                'Moving' : 'lane-keeping',
                                'Moving_towards' : 'lane-keeping',
                                'Moving_away' : 'lane-keeping'}


 ('Hazard_lights_on', 'In_outgoing_lane'),
  ('Hazard_lights_on', 'In_incoming_lane'), 
   ('Indicating_right', 'At_junction'), 
    ('Indicating_left', 'At_junction'),
# For vehicles 
vehicles_map = {('Hazard_lights_on', 'In_left_parking_area'):'stopped', 
                ('Moving_away', 'At_junction'):'lane-keeping',
                ('Indicating_left', 'In_vehicle_lane'):'lane-keeping',
                ('Moving_right', 'At_bus_stop'):'merge-right',
                ('Moving_towards', 'In_incoming_lane'):'lane-keeping', 
                ('Turning_right', 'In_incoming_lane'):'turn-right',
                ('Stopped', 'At_bus_stop'):'stopped',
                ('Moving', 'At_crossing'):'lane-keeping',  
                ('Revering', 'In_right_parking_area'):'reversing', 
                ('Breaking', 'At_bus_stop'):'braking',
                ('Turning_left', 'In_incoming_lane'):'turn-left',
                ('Moving_away', 'In_incoming_lane'):'lane-keeping', 
                ('Turning_right', 'In_vehicle_lane'):'turn-right', 
                ('Moving', 'In_right_parking_area'):'lane-keeping', 
                ('Indicating_right', 'In_outgoing_lane'):'lane-keeping',
                ('Turning_right', 'At_crossing'):'turn-right',
                ('Moving_right', 'In_outgoing_lane'):'merge-right', 
                ('Breaking', 'In_outgoing_lane'):'braking', 
                ('Moving_left', 'In_outgoing_lane'):'merge-left',
                ('Moving_away', 'In_vehicle_lane'):'lane-keeping',  
                ('Turning_right', 'In_right_parking_area'):'turn-right',
                ('Stopped', 'In_outgoing_lane'):'stopped', 
                ('Turning_left', 'In_vehicle_lane'):'turn-left',
                ('Breaking', 'In_left_parking_area'):'braking', 
                ('Moving_right', 'At_junction'):'merge-right', 
                ('Moving_left', 'At_junction'):'merge-left',
                ('Stopped', 'At_junction'):'stopped', 
                ('Stopped', 'On_left_pavement'):'stopped', 
                ('Breaking', 'At_junction'):'braking', 
                ('Hazard_lights_on', 'In_right_parking_area'):'stopped', 
                ('Moving_right', 'In_incoming_lane'):'merge-right', 
                ('Stopped', 'In_incoming_lane'):'stopped', 
                ('Moving_right', 'In_vehicle_lane'):'merge-right', 
                ('Indicating_right', 'In_vehicle_lane'):'lane-keeping', 
                ('Breaking', 'In_vehicle_lane'):'braking', 
                ('Moving_left', 'In_vehicle_lane'):'merge-left', 
                ('Indicating_left', 'In_outgoing_lane'):'lane-keeping', 
                ('Stopped', 'In_vehicle_lane'):'stopped', 
                ('Moving_left', 'In_right_parking_area'):'merge-left', 
                ('Moving', 'In_outgoing_lane'):'lane-keeping', 
                ('Breaking', 'In_right_parking_area'):'braking', 
                ('Moving_towards', 'In_left_parking_area'):'lane-keeping',  
                ('Moving_towards', 'In_outgoing_lane'):'lane-keeping',  
                ('Turning_right', 'In_outgoing_lane'):'turn-right', 
                ('Turning_right', 'In_left_parking_area'):'turn-right', 
                ('Moving', 'At_junction'):'lane-keeping', 
                ('Turning_left', 'In_outgoing_lane'):'turn-left', 
                ('Moving_away', 'In_outgoing_lane'):'lane-keeping',  
                ('Moving_towards', 'At_junction'):'lane-keeping',  
                ('Moving_towards', 'On_left_pavement'):'lane-keeping',  
                ('Turning_right', 'At_junction'):'turn-right', 
                ('Hazard_lights_on', 'At_bus_stop'):'stopped', 
                ('Revering', 'In_incoming_lane'):'reversing', 
                ('Turning_left', 'At_junction'):'turn-left',  
                ('Moving', 'In_incoming_lane'):'lane-keeping'}

 
ignore_objects = set(["Vehicle_traffic_light", "Other_traffic_light", "AV"])
vehicles_category = set(["Car", "Medium_vehicle", "Large_vehicle", "Bus", "Emergency_vehicle"])

def read_json(filename):
    with open(filename,'r') as json_file:
        data = json.load(json_file)
        return data

def save_labels_to_txt(labels, folder_path, file_name):
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{file_name}.json")
    
    with open(file_path, "w") as file:
        json.dump(labels, file)
    print(f"Saved: {file_path}")

        
        
def generate_intention_annotations(raw_annotations, prediction_annotations_path):
    actions = []
    tracks = []
    for ann in raw_annotations:
        files = sorted(os.listdir(ann))
        prediction_labels = {} 
        last_id = 1
        
        
        for file in files: 
            file_path = os.path.join(ann, file)
            if os.path.isfile(file_path) and file.endswith(".json"):
                data = read_json(file_path)
                frame_id = file.split('.')[0]
                
                for instance in data[0]['instances']: 
                    values = instance['classValues']
                    agent = values[0]['value']
                    location = values[1]['value']
                    action = values[2]['value']
                    intention = (action, location)
                    
                    if agent =="Unknown" or  agent in ignore_objects: continue
                    if agent=="Emergency vehicle": agent = "Emergency_vehicle"
                    if agent != "Pedestrian" and action == "Stopped" and "parking" in location: continue
                    
                    
                    if agent in vehicles_category:
                    #    continue
                    #if agent == "Small_motorised_vehicle":
                        actions.append(intention)
                    
                    tracks.append((instance["trackId"], agent))
                                        
                    if agent == 'Pedestrian':
                        intention = pedestrian_map[action]
                    elif agent == 'Cyclist':
                        intention = cyclist_map[action]
                    elif agent == 'Motorbike':
                        if intention in motorbike_ignore: continue
                        intention = motorbike_map[action]
                    elif agent == "Small_motorised_vehicle":
                        if intention in small_motorised_vehicles_ignore: continue
                        intention = small_motorised_vehicles_map[action]     
                    
                    # Extract and remap trackId
                    track_id = instance["trackId"]
                    if (track_id, agent) not in prediction_labels:
                        prediction_labels[(track_id, agent)] = {'object_id':last_id, 'class':agent, 'frames':[], 'bbox':[], 'intention':[]}
                        last_id += 1
                        
                    
                    # Extract bounding box points
                    points = instance["contour"]["points"]
                    bbox_left = min(point["x"] for point in points)
                    bbox_top = min(point["y"] for point in points)
                    bbox_right = max(point["x"] for point in points)
                    bbox_bottom = max(point["y"] for point in points)
                    
                    
                    prediction_labels[(track_id, agent)]['bbox'].append((bbox_left, bbox_top, bbox_right, bbox_bottom))
                    prediction_labels[(track_id, agent)]['frames'].append(int(frame_id))
                    prediction_labels[(track_id, agent)]['intention'].append(intention)

        objects_predictions = {pd['object_id']: {'class': pd['class'], 'frames': pd['frames'], 'bbox': pd['bbox'], 'intention':pd['intention']} for pd in prediction_labels.values() if len(pd['frames']) >= 20}
        objects_predictions = dict(sorted(objects_predictions.items()))
        
    print(set(actions))
        #print(a)
        
        #tracks = sorted(set(tracks))
        #tracks = Counter(t[0] for t in tracks)
        #print(tracks)
        #print(len(tracks))
        #print(len(objects_predictions.keys()))
        
        #print(a)
        #########################################################################
        # map (action, location) to intention
        
        
        ##for k,v in objects_predictions.items():
        #    if v['class'] == 'Small_motorised_vehicle':
        #        print(k,v)
            
        #print(a)
        
        #file_name = ann.split('/')[-1]
        #save_labels_to_txt(prediction_labels, prediction_annotations_path, file_name)
        

def main():
    raw_annotations_path = "../data/raw_annotations"
    prediction_annotations_path = "../data/annotations/intention_annotations"
    raw_annotations = [raw_annotations_path + '/' + f for f in os.listdir(raw_annotations_path)]
    generate_intention_annotations(raw_annotations, prediction_annotations_path)
    
if __name__ == '__main__':
    main()