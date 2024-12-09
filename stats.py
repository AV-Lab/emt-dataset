import json
import argparse
import os

def process_kitti_annotation(file_path, vehicles,av, unique_agents_vehicles, unique_agents_people,Av_agents,
            unknown_agents):
    """Process a single KITTI format annotation file"""
    bbox_count = 0
    
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # if len(parts) < 15:  # Basic validation
            #     continue
                
            frame_id = parts[0]
            track_id = parts[1]
            obj_class = parts[2]
        
            # Categorize the object
            if obj_class in ['Pedestrian', 'Cyclist']:
                unique_agents_people.add(track_id)
                bbox_count += 1
            elif obj_class in vehicles:
                unique_agents_vehicles.add(track_id)
                bbox_count += 1
            elif obj_class in av:
                Av_agents.add(track_id)
            else:
                # print(f"Unknown objects: {obj_class}: {file_path}")
                unknown_agents.add(track_id)
                
    return bbox_count

def main():
    p = argparse.ArgumentParser(description='compute the dataset statistics')
    p.add_argument('annot_dir', type=str, help='Annotations directory')
    args = p.parse_args()
    
    # Define vehicle classes
    vehicles = {
        'Car', 
        'Large_vehicle', 
        'Medium_vehicle',
        'Motorbike',
        'Bus',
        'Emergency_vehicle',
        'Small_motorised_vehicle'
    }
    av ={'AV'}
    bounding_box_count = 0
    unique_agents_vehicles = set()
    unique_agents_people = set()
    Av_agents = set()
    unknown_agents = set()
    
    # Process all annotation files
    annotation_files = [f for f in os.listdir(args.annot_dir) if f.endswith('.txt')]
    print(f"Processing annotation files: {annotation_files}")
    
    for ann_file in annotation_files:
        ann_path = os.path.join(args.annot_dir, ann_file)
        bbox_count = process_kitti_annotation(
            ann_path, 
            vehicles,
            av,
            unique_agents_vehicles,
            unique_agents_people,
            Av_agents,
            unknown_agents
        )
        bounding_box_count += bbox_count
    
    pd = len(unique_agents_people)
    vh = len(unique_agents_vehicles)
    
    print('\n-------------------------\nTotal number of bounding box annotations:', bounding_box_count)
    print('Total number of people (pedestrians and Cyclists):', pd)
    print('Total number of vehicles:', vh)
    print('Total number of unique agents:', pd + vh)


    
    # Additional statistics
    print('\n-------------------------\nDetailed Statistics:')
    if len(Av_agents) > 0:
        print('Number of AV agents:', len(Av_agents))
    if len(unknown_agents) > 0:
        print('Number of unclassified objects:', len(unknown_agents))
    print('Number of annotation files processed:', len(annotation_files))


if __name__ == '__main__':
    main()