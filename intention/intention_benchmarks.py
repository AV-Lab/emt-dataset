#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 20:12:11 2025

@author: nadya
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 20:09:57 2025

@author: nadya
"""


past = [10, 15, 20]
horizont = [10, 20, 30]


def main():
    annotations_path = "../data/annotations/prediction_annotations"
    raw_annotations = [raw_annotations_path + '/' + f for f in os.listdir(annotations_path)]
    generate_prediction_annotations(raw_annotations, prediction_annotations_path)
    
if __name__ == '__main__':
    main()